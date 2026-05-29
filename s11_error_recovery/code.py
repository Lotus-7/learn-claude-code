#!/usr/bin/env python3
"""
s11: 错误恢复 (Error Recovery) — 三条恢复路径 + 指数退避重试。

运行: python s11_error_recovery/code.py
依赖: pip install anthropic python-dotenv + .env 文件中配置 ANTHROPIC_API_KEY

相比 s10 的变更:
  - 在大模型(LLM)调用外层包裹 try/except，实现了三条错误恢复路径
  - 路径 1: max_tokens(输出被截断) -> 扩容策略(8K 升到 64K，首次不附加内容重试)，
            如果依旧截断，发送继续生成提示词 (最多续写 3 次)
  - 路径 2: prompt_too_long(上下文太长) -> 紧急压缩(reactive compact) -> 重试 (限 1 次)
  - 路径 3: 429/529(限流/过载) -> 附带抖动的指数退避重试机制 (最多重试 10 次)，
            连续收到 529 后退级使用备用模型 (fallback model)
  - 引入了处理瞬态错误的 with_retry 封装
  - RecoveryState 用于在整个循环中跟踪升级、压缩、529及模型降级的状态

ASCII 流程图:
  messages -> 提示词组装 -> 压缩并加载 -> [try] 调用大模型 [except] -> 执行工具 -> loop
                                                    |                 |
                                              停止原因判定        捕获错误类型
                                              max_tokens?      prompt_too_long? -> 压缩
                                              扩容 / 续写       429/529? -> 退避重试
                                                               其他? -> 打印日志并退出
"""

import os, subprocess, time, random, json
from pathlib import Path
from typing import Optional

try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
PRIMARY_MODEL = os.getenv("MODEL_ID", "deepseek-v4-pro")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL_ID")

# ── 常量配置 ──

ESCALATED_MAX_TOKENS = 64000
DEFAULT_MAX_TOKENS = 8000
MAX_RECOVERY_RETRIES = 3      # max_tokens 续写的最大次数
MAX_RETRIES = 10              # API 限流/错误最大重试次数
BASE_DELAY_MS = 500           # 基础重试延迟
MAX_CONSECUTIVE_529 = 3       # 连续 529 阈值，达到后降级模型
CONTINUATION_PROMPT = (
    "Output token limit hit. Resume directly — "
    "no apology, no recap. Pick up mid-thought."
)

# ── 提示词组装 (来自 s10, 保持一致) ──

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}

def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")
    return "\n\n".join(sections)

_last_context_key, _last_prompt = None, None

def get_system_prompt(context: dict) -> str:
    """带缓存机制获取系统提示词"""
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)

    loaded = ["identity", "tools", "workspace"]
    if context.get("memories"):
        loaded.append("memory")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt


# ── 基础工具 (未修改) ──

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: Optional[int] = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
]

TOOL_HANDLERS = {"bash": run_bash, "read_file": run_read, "write_file": run_write}


# ── s11 新增: 错误恢复机制 ──

class RecoveryState:
    """在代理循环中追踪故障恢复的状态"""
    def __init__(self):
        self.has_escalated = False               # 是否已提升了最大 token 限制
        self.recovery_count = 0                  # 续写尝试次数
        self.consecutive_529 = 0                 # 连续 529(过载) 错误次数
        self.has_attempted_reactive_compact = False # 是否已经尝试过紧急压缩
        self.current_model = PRIMARY_MODEL       # 当前使用的模型 (可能降级)


def retry_delay(attempt, retry_after=None):
    """附带随机抖动(jitter)的指数退避重试延迟算法。如果对方返回 Retry-After 则优先采用。"""
    if retry_after:
        return retry_after
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def with_retry(fn, state: RecoveryState):
    """
    针对瞬态错误(429限流/529过载)的指数退避重试封装。
    如果遇到非瞬态错误，直接抛出给外层捕获。
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_529 = 0  # 成功后重置 529 计数器
            return result
        except Exception as e:
            name = type(e).__name__
            msg = str(e).lower()

            # 429 请求超限 -> 触发指数退避
            if "ratelimit" in name.lower() or "429" in msg:
                delay = retry_delay(attempt)
                print(f"  \033[33m[429 rate limit] 重试 {attempt+1}/{MAX_RETRIES},"
                      f" 等待 {delay:.1f} 秒\033[0m")
                time.sleep(delay)
                continue

            # 529 服务过载 -> 触发指数退避，连续多次后尝试降级备用模型
            if "overloaded" in name.lower() or "529" in msg or "overloaded" in msg:
                state.consecutive_529 += 1
                if state.consecutive_529 >= MAX_CONSECUTIVE_529:
                    if FALLBACK_MODEL:
                        state.current_model = FALLBACK_MODEL
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 连续发生 x{MAX_CONSECUTIVE_529}]"
                              f" 切换至备用模型 {FALLBACK_MODEL}\033[0m")
                    else:
                        state.consecutive_529 = 0
                        print(f"  \033[31m[529 连续发生 x{MAX_CONSECUTIVE_529}]"
                              f" 未配置备用模型，继续重试\033[0m")
                delay = retry_delay(attempt)
                print(f"  \033[33m[529 overloaded] 重试 {attempt+1}/{MAX_RETRIES},"
                      f" 等待 {delay:.1f} 秒\033[0m")
                time.sleep(delay)
                continue

            # 非瞬态错误 -> 继续抛出供外部的 try/except 处理
            raise
    raise RuntimeError(f"已达到最大重试次数 ({MAX_RETRIES})")


def is_prompt_too_long_error(e: Exception) -> bool:
    """判定 API 错误是否为上下文长度超限"""
    msg = str(e).lower()
    return (("prompt" in msg and "long" in msg)
            or "prompt_is_too_long" in msg
            or "context_length_exceeded" in msg
            or "max_context_window" in msg)


def reactive_compact(messages: list) -> list:
    """紧急压缩机制 — 教学版为了简单仅保留最后 N 条消息。
    在真正的 Claude Code 中，这里会调用 LLM 生成摘要然后再继续。
    (因为在 s08/s09 中已经有常规的 LLM 摘要机制了，此处采用尾部截断即可)"""
    print("  \033[31m[reactive compact] 紧急修剪，仅保留最后 5 条消息\033[0m")
    tail = messages[-5:]
    return [{"role": "user",
             "content": "[Reactive compact] 早期对话已被修剪。请接着往下处理。"}, *tail]


# ── 上下文 ──

def update_context(context: dict, messages: list) -> dict:
    """根据真实状态更新运行上下文"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": list(TOOL_HANDLERS.keys()),
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# ── 代理循环 ──

def agent_loop(messages: list, context: dict):
    """主循环，包裹着错误恢复机制"""
    system = get_system_prompt(context)
    state = RecoveryState()
    max_tokens = DEFAULT_MAX_TOKENS

    while True:
        # ── 路径 3：with_retry 会处理 429/529 瞬态错误，外部负责其余错误 ──
        try:
            response = with_retry(
                lambda mt=max_tokens, mdl=state.current_model:
                    client.messages.create(
                        model=mdl, system=system, messages=messages,
                        tools=TOOLS, max_tokens=mt),
                state)
        except Exception as e:
            # 路径 2: prompt_too_long -> 触发一次紧急截断
            if is_prompt_too_long_error(e):
                if not state.has_attempted_reactive_compact:
                    messages[:] = reactive_compact(messages)
                    state.has_attempted_reactive_compact = True
                    continue
                print("  \033[31m[unrecoverable] 截断后长度依旧超限，无法继续\033[0m")
                messages.append({"role": "assistant", "content": [
                    {"type": "text",
                     "text": "[Error] Context too large, cannot continue."}]})
                return

            # 其他无法恢复的致命错误
            name = type(e).__name__
            print(f"  \033[31m[unrecoverable] {name}: {str(e)[:100]}\033[0m")
            messages.append({"role": "assistant", "content": [
                {"type": "text", "text": f"[Error] {name}: {str(e)[:200]}"}]})
            return

        # ── 路径 1: max_tokens(输出长度达到限制) -> 升级限额 或 进行续写 ──
        if response.stop_reason == "max_tokens":
            # 首次升级：不追加被截断的内容，直接使用 64K 的上限发起同样的请求
            if not state.has_escalated:
                max_tokens = ESCALATED_MAX_TOKENS
                state.has_escalated = True
                print(f"  \033[33m[max_tokens] 升级 token 限制"
                      f" {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}\033[0m")
                continue
            
            # 64K 也被截断了：保留截断结果，并附加一条提示词让模型续写
            messages.append({"role": "assistant", "content": response.content})
            if state.recovery_count < MAX_RECOVERY_RETRIES:
                messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                state.recovery_count += 1
                print(f"  \033[33m[max_tokens] 请求续写"
                      f" {state.recovery_count}/{MAX_RECOVERY_RETRIES}\033[0m")
                continue
            print("  \033[31m[max_tokens] 达到最大续写次数限制\033[0m")
            return

        # 正常结束：将助手的回答追加到历史中
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        # ── 工具执行 ──
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result",
                            "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})

        # 回合结束，更新上下文并刷新系统提示词
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s11: 错误恢复 (Error Recovery) — 提供三条强壮的容错及恢复机制")
    print("输入问题，按回车发送。输入 q 退出。\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms11 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        turn_start = len(history)
        history.append({"role": "user", "content": query})
        
        agent_loop(history, context)
        
        context = update_context(context, history)
        for msg in history[turn_start:]:
            if msg.get("role") != "assistant":
                continue
            for block in msg["content"]:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
