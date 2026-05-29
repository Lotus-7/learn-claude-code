#!/usr/bin/env python3
"""
s03_permission.py - 权限系统

在工具执行前插入了三个拦截关卡：

    第一关：硬性黑名单（rm -rf /, sudo 等绝对禁止）
    第二关：规则匹配（写入工作区外部？包含潜在破坏性命令？）
    第三关：人工审批（暂停程序，等待用户输入确认）

    +-------+    +--------+    +--------+    +--------+    +------+
    | 工具  | -> | 第一关 | -> | 第二关 | -> | 第三关 | -> | 执行 |
    | 调用  |    | 拦截？ |    | 匹配？ |    | 允许？ |    |      |
    +-------+    +--------+    +--------+    +--------+    +------+
         |            |             |             |
         v            v             v             v
      (正常)      (直接拦截)    (询问用户)    (用户拒绝？)

核心循环 (agent_loop) 中仅增加了一行核心拦截代码：

    if not check_permission(block):
        continue

基于 s02（多工具支持）构建。用法：

    python s03_permission/code.py
    需要：pip install anthropic python-dotenv 并在 .env 中配置 ANTHROPIC_API_KEY
"""

import os, subprocess
from pathlib import Path
from typing import Optional

# 尝试导入 readline 模块以增强终端交互体验（支持上下键查看历史等）
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

# 加载环境变量（覆盖已有变量）
load_dotenv(override=True)
# 如果配置了自定义的 Base URL，则移除默认的 Token（通常用于兼容其他 API 提供商）
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 获取当前工作目录
WORKDIR = Path.cwd()
# 初始化 Anthropic 客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
# 获取模型 ID，默认为 deepseek-v4-pro
MODEL = os.getenv("MODEL_ID", "deepseek-v4-pro")

# 设定系统提示词，告诉 Agent 所有破坏性操作都需要用户批准
SYSTEM = f"You are a coding agent at {WORKDIR}. All destructive operations require user approval."


# ═══════════════════════════════════════════════════════════
#  FROM s02 (unchanged): Tool Implementations
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """路径安全校验：确保操作的路径不会逃逸出当前工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_bash(command: str) -> str:
    """执行 Bash 命令的工具函数（安全检查已移至专门的权限模块）。"""
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"


def run_read(path: str, limit: Optional[int] = None) -> str:
    """读取文件内容的工具函数，支持限制读取行数。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... ({len(lines) - limit} more lines)"]
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def run_write(path: str, content: str) -> str:
    """将内容写入文件的工具函数，会自动创建不存在的父目录。"""
    try:
        file_path = safe_path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error: {e}"


def run_edit(path: str, old_text: str, new_text: str) -> str:
    """文本替换工具：在文件中将精确匹配的 old_text 替换为 new_text。"""
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
    """文件搜索工具：使用 glob 模式匹配工作目录下的文件。"""
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
#  FROM s02 (unchanged): Tool Definitions & Dispatch
# ═══════════════════════════════════════════════════════════

# 定义提供给大语言模型的工具列表及其输入 Schema
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

# 建立工具名称与实际 Python 函数的映射字典
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  NEW in s03: Three-Gate Permission Pipeline
# ═══════════════════════════════════════════════════════════

# Gate 1: Hard deny list — always forbidden
# 第一关：硬性黑名单（绝对禁止的危险命令，包含关键字直接拦截）
DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]

def check_deny_list(command: str) -> Optional[str]:
    """检查命令是否触发了硬性黑名单"""
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None


# Gate 2: Rule matching — context-dependent checks
# 第二关：规则匹配（需要根据上下文或参数动态判断的操作）
PERMISSION_RULES = [
    {"tools": ["write_file", "edit_file"],
     # 拦截试图写到工作目录之外的操作
     "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
     "message": "Writing outside workspace"},
    {"tools": ["bash"],
     # 拦截潜在的破坏性命令（如删除文件、修改系统配置、更改权限等）
     "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
     "message": "Potentially destructive command"},
]

def check_rules(tool_name: str, args: dict) -> Optional[str]:
    """根据定义好的规则库检查工具调用是否需要用户确认"""
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# Gate 3: User approval — wait for confirmation after rule match
# 第三关：用户审批（当触发了第二关的规则时，暂停并询问用户）
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    """终端交互：打印警告信息，等待用户输入 y 或 n"""
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"


# Pipeline: all three gates chained
def check_permission(block) -> bool:
    """
    权限检查流水线主入口：
    1. 针对 bash 命令检查硬性黑名单
    2. 针对所有工具检查动态规则
    3. 如果触发规则，则弹出确认框让用户审批
    """
    # 针对 bash 进行第一关：硬性拦截
    if block.name == "bash":
        reason = check_deny_list(block.input.get("command", ""))
        if reason:
            print(f"\n\033[31m⛔ {reason}\033[0m")
            return False
            
    # 进行第二关：规则匹配
    reason = check_rules(block.name, block.input)
    if reason:
        # 触发规则，进入第三关：人工审批
        decision = ask_user(block.name, block.input, reason)
        if decision == "deny":
            return False
            
    # 一路绿灯，放行
    return True


# ═══════════════════════════════════════════════════════════
#  agent_loop — same as s02, with check_permission() inserted
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """
    Agent 核心循环（新增权限检查版）：
    1. 获取模型回复
    2. 解析工具调用
    3. **拦截检查**：调用 check_permission 流水线
    4. 执行工具或返回拒绝信息
    """
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            print(f"\033[36m> {block.name}\033[0m")

            # s03 change: run through permission pipeline before executing
            # 【核心修改点】在执行前进行权限流水线检查
            if not check_permission(block):
                # 如果被拒绝，把拒绝信息告诉模型，让它知道这条路走不通
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": "Permission denied."})
                continue

            handler = TOOL_HANDLERS.get(block.name)
            output = handler(**block.input) if handler else f"Unknown: {block.name}"
            print(str(output)[:200])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s03: Permission")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    # 简单的交互式 REPL 循环
    while True:
        try:
            query = input("\033[36ms03 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("q", "exit", ""):
            break
        history.append({"role": "user", "content": query})
        agent_loop(history)
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
