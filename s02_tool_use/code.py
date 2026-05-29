#!/usr/bin/env python3
"""
s02: Tool Use — 在 s01 基础上新增 4 个工具 + 分发映射。

运行: python s02_tool_use/code.py
需要: pip install anthropic python-dotenv + .env 中配置 ANTHROPIC_API_KEY

本文件 = s01 的全部代码 + 以下新增:
  + run_read / run_write / run_edit / run_glob 四个工具实现
  + TOOL_HANDLERS 分发映射（替代 s01 中硬编码的 run_bash 调用）
  + safe_path 路径安全校验

循环本身（agent_loop）与 s01 完全一致。
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

# 设定系统提示词，告诉 Agent 它的身份和任务
SYSTEM = f"You are a coding agent at {WORKDIR}. Use tools to solve tasks. Act, don't explain."


# ═══════════════════════════════════════════════════════════
#  FROM s01 (unchanged)
# ═══════════════════════════════════════════════════════════

def run_bash(command: str) -> str:
    """执行 Bash 命令的工具函数，带有简单的安全检查。"""
    # 简单的危险命令拦截
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked"
    try:
        # 在指定工作目录执行命令，捕获输出
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=120)
        # 合并标准输出和标准错误
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度为 50000 字符，避免撑爆上下文
        return out[:50000] if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  NEW in s02: 4 个新工具
# ═══════════════════════════════════════════════════════════

def safe_path(p: str) -> Path:
    """路径安全校验：确保操作的路径不会逃逸出当前工作目录。"""
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


def run_read(path: str, limit: Optional[int] = None) -> str:
    """读取文件内容的工具函数，支持限制读取行数。"""
    try:
        lines = safe_path(path).read_text().splitlines()
        # 如果指定了 limit 且文件行数超过 limit，则截断并提示省略的行数
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
        # 确保被替换的文本存在于文件中
        if old_text not in text:
            return f"Error: text not found in {path}"
        # 只替换第一次出现的匹配项
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
            # 同样需要进行路径安全校验
            if (WORKDIR / match).resolve().is_relative_to(WORKDIR):
                results.append(match)
        return "\n".join(results) if results else "(no matches)"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具定义（s01 只有一个 bash，现在扩展到 5 个）
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

# ═══════════════════════════════════════════════════════════
#  NEW in s02: 工具分发映射（s01 是硬编码 run_bash，现在改为查表）
# ═══════════════════════════════════════════════════════════

# 建立工具名称与实际 Python 函数的映射字典
TOOL_HANDLERS = {
    "bash": run_bash, "read_file": run_read, "write_file": run_write,
    "edit_file": run_edit, "glob": run_glob,
}


# ═══════════════════════════════════════════════════════════
#  agent_loop — 与 s01 结构完全一致，只改了工具执行那部分
#  s01: output = run_bash(block.input["command"])
#  s02: output = TOOL_HANDLERS[block.name](**block.input)
# ═══════════════════════════════════════════════════════════

def agent_loop(messages: list):
    """
    Agent 核心循环：
    1. 调用大模型获取回复
    2. 如果模型决定使用工具，则解析并执行工具
    3. 将工具执行结果追加到消息列表中，继续循环
    4. 直到模型决定停止（stop_reason != "tool_use"）
    """
    while True:
        # 调用大语言模型 API
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        # 将助手的回复加入历史消息
        messages.append({"role": "assistant", "content": response.content})

        # 如果停止原因不是使用了工具，说明任务已完成，跳出循环
        if response.stop_reason != "tool_use":
            return

        results = []
        # 遍历返回内容中的每一个块
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m> {block.name}\033[0m")
                # 从映射表中获取对应的处理函数
                handler = TOOL_HANDLERS.get(block.name)
                # 执行工具函数，传入模型提供的参数
                output = handler(**block.input) if handler else f"Unknown: {block.name}"
                print(str(output)[:200])
                # 记录工具调用的结果
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})

        # 将工具结果作为用户的回复传递给模型
        messages.append({"role": "user", "content": results})


if __name__ == "__main__":
    print("s02: Tool Use — 在 s01 基础上加了 4 个工具")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    # 简单的交互式 REPL 循环
    while True:
        try:
            query = input("\033[36ms02 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        # 处理退出命令
        if query.strip().lower() in ("q", "exit", ""):
            break
        # 将用户输入加入历史消息
        history.append({"role": "user", "content": query})
        # 启动 Agent 处理任务
        agent_loop(history)
        # 打印出模型最后回复的纯文本内容（跳过 tool_use 块）
        for block in history[-1]["content"]:
            if getattr(block, "type", None) == "text":
                print(block.text)
        print()
