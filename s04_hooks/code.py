#!/usr/bin/env python3
"""
s04: 钩子系统 (Hooks) — 将扩展逻辑从主循环中移出，交给钩子处理。

  用户输入问题
       │
       ▼
  ┌──────────────────┐
  │ UserPromptSubmit │ ── 在调用大语言模型(LLM)前触发钩子
  └────────┬─────────┘
           ▼
  ┌────────────┐     ┌─────────────────────────────┐
  │  messages  │────▶│  LLM (stop_reason=tool_use?)│
  └────────────┘     │   No ──▶ 触发停止钩子 ──▶ 退出 │
                     │   Yes ──▶ 执行 tool_use 模块 ──┐ │
                     └────────────────────────────┘ │
                                                    ▼
                                          ┌──────────────────┐
                                          │ trigger_hooks()   │
                                          │  PreToolUse (工具调用前): │
                                          │   permission_hook (权限校验) │
                                          │   log_hook (日志记录) │
                                          └───────┬──────────┘
                                                  │ (未被拦截)
                                          ┌───────▼──────────┐
                                          │ TOOL_HANDLERS[x]  │
                                          └───────┬──────────┘
                                                  │
                                          ┌───────▼──────────┐
                                          │ trigger_hooks()   │
                                          │  PostToolUse (工具调用后):│
                                          │   large_output (大输出警告)│
                                          └───────┬──────────┘
                                                  │
                                          results ──▶ 返回到 messages 列表中

相比 s03 的变更:
  + HOOKS 注册表 (事件 -> 回调函数列表)
  + register_hook() / trigger_hooks() (注册/触发钩子)
  + context_inject_hook (处理 UserPromptSubmit 事件)
  + permission_hook, log_hook (处理 PreToolUse 事件)
  + large_output_hook (处理 PostToolUse 事件)
  + summary_hook (处理 Stop 事件)
  - 移除了循环体中的 check_permission() 
    (逻辑被移动到 permission_hook 中，通过 PreToolUse 触发)

运行: python s04_hooks/code.py
依赖: pip install anthropic python-dotenv + .env 文件中配置 ANTHROPIC_API_KEY
"""

import os, subprocess
from pathlib import Path
from typing import Optional

# 尝试配置 readline 以优化终端输入体验
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(override=True)
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

WORKDIR = Path.cwd()
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "deepseek-v4-pro")

# 系统提示词，定义 AI 的角色和行为准则
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ═══════════════════════════════════════════════════════════
#  来自 s02-s03 (未修改): 工具的具体实现
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """确保路径安全，防止目录穿越"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path

def run_bash(command: str) -> str:
    """执行 Bash 命令"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"

def run_read(path: str, limit: Optional[int] = None) -> str:
    """读取文件内容"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"

def run_write(path: str, content: str) -> str:
    """将内容写入文件"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"

def run_edit(path: str, old_text: str, new_text: str) -> str:
    """替换文件中的特定文本"""
    try:
        file_path = safe_path(path)
        text = file_path.read_text()
        if old_text not in text:
            return f"Error: text not found in {path}"
        file_path.write_text(text.replace(old_text, new_text, 1))
        return f"Edited {path}"
    except Exception as e:
        return f"Error: {e}"

def run_glob(pattern: str) -> str:
    """根据 glob 模式查找文件"""
    import glob as g
    try:
        results = []
        for match in g.glob(pattern, root_dir=WORKDIR):
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"

# 定义给大模型使用的工具列表
TOOLS = [
    {"name": "bash", "description": "Run a shell command.",
     "input_schema": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}},
    {"name": "read_file", "description": "Read file contents.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["path"]}},
    {"name": "write_file", "description": "Write content to a file.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}},
    {"name": "edit_file", "description": "Replace exact text in a file once.",
     "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}}, "required": ["path", "old_text", "new_text"]}},
    {"name": "glob", "description": "Find files matching a glob pattern.",
     "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
]

# 工具名称到实际函数的映射
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  s04 新增: 钩子系统 (Hook System)
# ═══════════════════════════════════════════════════════════

# 定义钩子事件注册表
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    """注册一个回调函数到指定的钩子事件"""
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    """触发指定事件的所有钩子函数。如果任意钩子返回非空值，则提前返回该值。"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:  # 如果钩子返回了结果，说明它想要拦截/阻断当前操作
            return result
    return None


# 将 s03 中的权限检查逻辑重构为钩子形式
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]
DESTRUCTIVE = ["rm ", "> /etc/", "chmod 777"]

def permission_hook(block):
    """PreToolUse 钩子: 在执行工具前进行权限和安全校验"""
    if block.name == "bash":
        for pattern in DENY_LIST:
            if pattern in block.input.get("command", ""):
                print(f"\n\033[31m⛔ 已拦截: '{pattern}'\033[0m")
                return "Permission denied by deny list"
        for kw in DESTRUCTIVE:
            if kw in block.input.get("command", ""):
                print(f"\n\033[33m⚠  检测到潜在的破坏性命令\033[0m")
                print(f"   工具: {block.name}({block.input})")
                choice = input("   是否允许? [y/N] ").strip().lower()
                if choice not in ("y", "yes"):
                    return "Permission denied by user"
    if block.name in ("write_file", "edit_file"):
        path = block.input.get("path", "")
        if not (WORKDIR / path).resolve().is_relative_to(WORKDIR):
            print(f"\n\033[33m⚠  尝试在工作区外写入文件\033[0m")
            print(f"   工具: {block.name}({block.input})")
            choice = input("   是否允许? [y/N] ").strip().lower()
            if choice not in ("y", "yes"):
                return "Permission denied by user"
    return None

def log_hook(block):
    """PreToolUse 钩子: 记录每次的工具调用"""
    args_preview = str(list(block.input.values())[:2])[:60]
    print(f"\033[90m[HOOK] {block.name}({args_preview})\033[0m")
    return None

def large_output_hook(block, output):
    """PostToolUse 钩子: 对超大输出结果发出警告"""
    if len(str(output)) > 100000:
        print(f"\033[33m[HOOK] ⚠ {block.name} 的输出过大: {len(str(output))} 字符\033[0m")
    return None

def context_inject_hook(query: str):
    """UserPromptSubmit 钩子: 在将用户输入传给 LLM 之前记录工作目录"""
    print(f"\033[90m[HOOK] UserPromptSubmit: 当前工作目录为 {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    """Stop 钩子: 在会话退出前打印工具调用统计"""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: 本次会话共调用了 {tool_count} 次工具\033[0m")
    return None

# 注册所有定义好的钩子
register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("PostToolUse", large_output_hook)
register_hook("Stop", summary_hook)


# ═══════════════════════════════════════════════════════════
#  主代理循环 (agent_loop)
#  s04 特色: 没有硬编码的权限检查，全部由 PreToolUse 钩子接管
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """核心 AI 代理循环"""
    while True:
        # 发送请求给 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        # 如果 LLM 不打算使用工具，触发 Stop 钩子并结束
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 触发 PreToolUse 钩子 (进行权限校验、日志记录等)
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            # 执行工具
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

            # 触发 PostToolUse 钩子
            trigger_hooks("PostToolUse", block, output)

            # 将工具结果保存起来
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        # 将所有的工具结果传回给 LLM 进行下一步决策
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s04: 钩子系统 (Hooks) — 将扩展逻辑移至钩子，保持主循环纯净")
    print("请输入问题并按回车。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms04 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        # 触发用户输入钩子
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        
        # 启动 Agent 循环
        agent_loop(history)
        
        # 打印 AI 最后的回复文本
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
