#!/usr/bin/env python3
"""
s05: 任务列表 (TodoWrite) — 在 s04 钩子的基础上添加任务规划工具。

  +---------+      +-------+      +------------------+
  |  用户    | ---> |  LLM  | ---> | TOOL_HANDLERS    |
  |  输入    |      |       |      |  bash            |
  +---------+      +---+---+      |  read_file       |
                        ^         |  write_file      |
                        | 返回结果 |  edit_file       |
                        +---------+  glob            |
                                      todo_write ← 新增工具
                                   +------------------+
                                        |
                                  在内存中维护 current_todos
                                        |
                        如果 rounds_since_todo >= 3 (即太久没更新任务了):
                          自动注入 <reminder> 提醒 AI 更新任务

相比 s04 的变更:
  + 新增 todo_write 工具及对应的 run_todo_write() 实现
  + 唠叨提醒机制 (如果在 3 轮交互内未更新任务，则注入提醒)
  + SYSTEM 系统提示词中增加了“执行前先规划”的引导
  + 在 agent_loop 中添加了 rounds_since_todo 计数器
  主循环结构未变: 新工具通过 TOOL_HANDLERS 自动分发。

运行: python s05_todo_write/code.py
依赖: pip install anthropic python-dotenv + .env 文件中配置 ANTHROPIC_API_KEY
"""

import os, subprocess
from pathlib import Path
from typing import Optional

# 尝试配置 readline 以优化终端输入体验
try:
    import readline
    readline.parse_and_bind('set bind-tty-special-chars off')
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
CURRENT_TODOS: list[dict] = []

# s05 变更: 系统提示词中增加了任务规划的引导
SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Before starting any multi-step task, use todo_write to plan your steps. "
    "Update status as you go."
)


# ═══════════════════════════════════════════════════════════
#  来自 s02-s04 (未修改): 基础工具的具体实现
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
    """写入文件内容"""
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


# ═══════════════════════════════════════════════════════════
#  s05 新增: todo_write 工具 — 仅用于规划任务，不执行实际操作
# ═══════════════════════════════════════════════════════════

def run_todo_write(todos: list) -> str:
    """管理和写入任务列表"""
    global CURRENT_TODOS
    # 校验必须的字段
    for i, t in enumerate(todos):
        if "content" not in t or "status" not in t:
            return f"Error: todos[{i}] missing 'content' or 'status'"
        if t["status"] not in ("pending", "in_progress", "completed"):
            return f"Error: todos[{i}] has invalid status '{t['status']}'"
    
    CURRENT_TODOS = todos
    lines = ["\n\033[33m## 当前任务列表\033[0m"]
    for t in CURRENT_TODOS:
        icon = {"pending": " ", "in_progress": "\033[36m▸\033[0m", "completed": "\033[32m✓\033[0m"}[t["status"]]
        lines.append(f"  [{icon}] {t['content']}")
    print("\n".join(lines))
    return f"Updated {len(CURRENT_TODOS)} tasks"

# 定义工具列表
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
    # s05 新增工具: todo_write
    {"name": "todo_write", "description": "Create and manage a task list for your current coding session.",
     "input_schema": {"type": "object", "properties": {"todos": {"type": "array", "items": {"type": "object", "properties": {"content": {"type": "string"}, "status": {"type": "string", "enum": ["pending", "in_progress", "completed"]}}, "required": ["content", "status"]}}}, "required": ["todos"]}},
]

TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob, "todo_write": run_todo_write,
}


# ═══════════════════════════════════════════════════════════
#  来自 s04 (未修改): 钩子系统
# ═══════════════════════════════════════════════════════════

HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}

def register_hook(event: str, callback):
    """注册钩子函数"""
    HOOKS[event].append(callback)

def trigger_hooks(event: str, *args):
    """触发特定事件的钩子"""
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None

# s04 权限黑名单
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if="]

def permission_hook(block):
    """PreToolUse: 黑名单权限校验"""
    if block.name == "bash":
        for p in DENY_LIST:
            if p in block.input.get("command", ""):
                print(f"\n\033[31m⛔ 已拦截: '{p}'\033[0m")
                return "Permission denied"
    return None

def log_hook(block):
    """PreToolUse: 记录工具调用日志"""
    print(f"\033[90m[HOOK] {block.name}\033[0m")
    return None

def context_inject_hook(query: str):
    """UserPromptSubmit: 输出当前工作目录日志"""
    print(f"\033[90m[HOOK] UserPromptSubmit: 当前工作目录为 {WORKDIR}\033[0m")
    return None

def summary_hook(messages: list):
    """Stop: 输出本次调用的统计信息"""
    tool_count = sum(1 for m in messages
                     for b in (m.get("content") if isinstance(m.get("content"), list) else [])
                     if isinstance(b, dict) and b.get("type") == "tool_result")
    print(f"\033[90m[HOOK] Stop: 本次会话共调用了 {tool_count} 次工具\033[0m")
    return None

register_hook("UserPromptSubmit", context_inject_hook)
register_hook("PreToolUse", permission_hook)
register_hook("PreToolUse", log_hook)
register_hook("Stop", summary_hook)


# ═══════════════════════════════════════════════════════════
#  主代理循环 (agent_loop) — 与 s04 基本相同，新增唠叨提醒
# ═══════════════════════════════════════════════════════════

# 用于跟踪多久没有更新 Todo 了
rounds_since_todo = 0

def agent_loop(messages: list):
    """核心 AI 代理循环"""
    global rounds_since_todo
    while True:
        # s05 新增: 唠叨提醒机制 — 如果连续 3 轮未更新任务，则在历史中注入提醒
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                             "content": "<reminder>Update your todos.</reminder>"})
            rounds_since_todo = 0

        # 调用 LLM
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force:
                messages.append({"role": "user", "content": force})
                continue
            return

        rounds_since_todo += 1
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            # 触发执行前钩子
            blocked = trigger_hooks("PreToolUse", block)
            if blocked:
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": str(blocked)})
                continue

            # 执行具体工具
            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"

            # 触发执行后钩子
            trigger_hooks("PostToolUse", block, output)

            # s05 变更: 如果调用了 todo_write 工具，则重置唠叨计数器
            if block.name == "todo_write":
                rounds_since_todo = 0

            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s05: TodoWrite 工具 — 执行前先规划，太久不更新会被提醒")
    print("请输入问题并按回车。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms05 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        
        trigger_hooks("UserPromptSubmit", query)
        history.append({"role": "user", "content": query})
        
        agent_loop(history)
        
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
