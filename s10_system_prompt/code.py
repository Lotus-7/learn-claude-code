#!/usr/bin/env python3
"""
s10: 动态系统提示词 (System Prompt) — 带有缓存机制的运行时提示词组装。

运行: python s10_system_prompt/code.py
依赖: pip install anthropic python-dotenv + .env 文件中配置 ANTHROPIC_API_KEY

相比 s09 的变更:
  - PROMPT_SECTIONS: 采用基于主题的提示词片段字典来维护内容
  - assemble_system_prompt(context): 基于真实的运行状态进行提示词组装
  - get_system_prompt(context): 利用 json.dumps 提供稳定的缓存，避免冗余组装
  - agent_loop 中使用 get_system_prompt(context) 代替硬编码的 SYSTEM 变量

记忆功能只有在 .memory/MEMORY.md 文件切实存在时才加载 (基于真实状态而非写死的关键字)。
"""

import os, subprocess, json
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
MODEL = os.getenv("MODEL_ID", "deepseek-v4-pro")


# ── 提示词分段定义 ──

PROMPT_SECTIONS = {
    "identity": "You are a coding agent. Act, don't explain.",
    "tools": "Available tools: bash, read_file, write_file.",
    "workspace": f"Working directory: {WORKDIR}",
    "memory": "Relevant memories are injected below when available.",
}


def assemble_system_prompt(context: dict) -> str:
    """基于当前的运行上下文，选择性拼接所需的提示词片段。"""
    sections = []

    # 始终加载的部分 — 身份、工具、工作目录
    sections.append(PROMPT_SECTIONS["identity"])
    sections.append(PROMPT_SECTIONS["tools"])
    sections.append(PROMPT_SECTIONS["workspace"])

    # 条件加载部分 — 仅当 MEMORY.md 存在且有内容时才注入记忆配置
    memories = context.get("memories", "")
    if memories:
        sections.append(f"Relevant memories:\n{memories}")

    return "\n\n".join(sections)


# 用于缓存的状态变量
_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    """带缓存机制的获取系统提示词的封装 — 仅当 context 改变时才重新组装。

    使用 json.dumps 获得确定性的序列化结果，不用 Python 原生的 hash()
    (因为 hash 存在进程级随机性且不支持嵌套字典/列表)。
    此处的缓存避免了同一进程内多余的字符串拼接操作。
    (在真实的 Claude Code 实现中，还会利用固定顺序 + SYSTEM_PROMPT_DYNAMIC_BOUNDARY 来保护 API 层面的 Prompt Caching)。
    """
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    
    # 缓存命中，直接返回
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


# ── 工具定义 ──

def safe_path(p: str) -> Path:
    """路径安全检查"""
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


# ── 上下文收集 ──

def update_context(context: dict, messages: list) -> dict:
    """从真实的运行状态提取上下文: 当前开启的工具、是否能读取到记忆索引等。"""
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


# ── 主代理循环 ──

def agent_loop(messages: list, context: dict):
    """主循环 — 使用动态组装的系统提示词，替代全局的硬编码 SYSTEM。"""
    system = get_system_prompt(context)
    while True:
        response = client.messages.create(
            model=MODEL, system=system, messages=messages,
            tools=TOOLS, max_tokens=8000)
        messages.append({"role": "assistant", "content": response.content})
        
        if response.stop_reason != "tool_use":
            return

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

        # 每次工具调用结束，重新评估环境状态并更新上下文和系统提示词
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s10: 动态提示词 (system prompt) — 运行时按需组装")
    print("输入问题，按回车发送。输入 q 退出。\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms10 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history, context)
        context = update_context(context, history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
