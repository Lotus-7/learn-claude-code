#!/usr/bin/env python3
"""
s01_agent_loop.py - 核心代理循环 (The Agent Loop)

这里展示了一个 AI 编程代理 (Coding Agent) 最核心的秘密，只需一个模式即可概括：

    while 停止原因 == "调用工具 (tool_use)":
        回复 = 大模型(历史消息, 工具列表)
        执行被调用的工具
        将工具执行结果追加到消息中

    +----------+      +-------+      +---------+
    |   用户   | ---> | 大模型 | ---> | 执行工具 |
    |  Prompt  |      | (LLM) |      | (Tool)  |
    +----------+      +---+---+      +----+----+
                          ^               |
                          |   工具结果    |
                          +---------------+
                          (循环继续)

这就是最核心的循环：将工具执行的结果不断喂给模型，
直到模型认为任务完成，决定停止调用工具为止。
生产环境中的 Agent 只是在这个基础上叠加了策略、钩子函数和生命周期控制。

使用方法:
    pip install anthropic python-dotenv
    配置好 .env 文件后运行: python s01_agent_loop/code.py
"""

import os
import subprocess

try:
    import readline
    # macOS 的 libedit 在处理中文输入时有退格问题，这四行配置可以修复它
    readline.parse_and_bind('set bind-tty-special-chars off')
    readline.parse_and_bind('set input-meta on')
    readline.parse_and_bind('set output-meta on')
    readline.parse_and_bind('set convert-meta off')
except ImportError:
    pass

from anthropic import Anthropic
from dotenv import load_dotenv

# 加载 .env 环境变量文件
load_dotenv(override=True)

# 如果配置了自定义的 Base URL，则移除默认的 AUTH_TOKEN 避免冲突
if os.getenv("ANTHROPIC_BASE_URL"):
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)

# 初始化客户端
client = Anthropic(base_url=os.getenv("ANTHROPIC_BASE_URL"))
MODEL = os.getenv("MODEL_ID", "deepseek-v4-pro")

# 设定系统提示词 (System Prompt)，告诉模型它的身份和能做的事
SYSTEM = f"You are a coding agent at {os.getcwd()}. Use bash to solve tasks. Act, don't explain."

# ── 1. 定义工具：这里我们只提供一个 Bash 命令行工具 ────────────────────────────
TOOLS = [{
    "name": "bash",
    "description": "运行 Shell 命令 (Run a shell command).",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]


# ── 2. 执行工具：实际运行命令的函数 ────────────────────────────────────────
def run_bash(command: str) -> str:
    # 拦截危险命令，防止破坏系统
    dangerous = ["rm -rf /", "sudo", "shutdown", "reboot", "> /dev/"]
    if any(d in command for d in dangerous):
        return "Error: Dangerous command blocked (已拦截危险命令)"
    try:
        # 执行命令并捕获输出，超时时间设置为 120 秒
        r = subprocess.run(command, shell=True, cwd=os.getcwd(),
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        # 限制输出长度，防止大段日志撑爆模型上下文窗口
        return out[:50000] if out else "(no output / 无输出)"
    except subprocess.TimeoutExpired:
        return "Error: Timeout (120s) (执行超时)"
    except (FileNotFoundError, OSError) as e:
        return f"Error: {e}"


# ── 3. 核心模式：不断调用工具直到模型停止的 while 循环 ──────────────────────
def agent_loop(messages: list):
    while True:
        # 请求大模型
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )

        # 将助手的回复追加到对话历史中
        messages.append({"role": "assistant", "content": response.content})

        # 如果模型没有选择调用工具，说明它觉得任务完成了，直接退出循环
        if response.stop_reason != "tool_use":
            return

        # 遍历模型的回复，提取并执行所有的工具调用请求
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"\033[33m$ {block.input['command']}\033[0m")
                output = run_bash(block.input["command"])
                print(output[:200]) # 仅在控制台打印前 200 个字符
                
                # 记录工具执行的结果
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })

        # 将收集到的所有工具执行结果喂回给模型，继续下一轮循环
        messages.append({"role": "user", "content": results})


# ── 4. 程序入口：与用户交互的命令行界面 ──────────────────────────────────────────
if __name__ == "__main__":
    print("s01: 核心代理循环 (Agent Loop)")
    print("输入问题，回车发送。输入 q 退出。\n")

    history = []
    while True:
        try:
            query = input("\033[36ms01 >> \033[0m")
        except (EOFError, KeyboardInterrupt):
            break
        
        # 退出条件判断
        if query.strip().lower() in ("q", "exit", ""):
            break
            
        # 记录用户输入并启动 Agent 循环
        history.append({"role": "user", "content": query})
        agent_loop(history)
        
        # 循环结束后，打印模型最终给出的文本回复
        response_content = history[-1]["content"]
        if isinstance(response_content, list):
            for block in response_content:
                if getattr(block, "type", None) == "text":
                    print(block.text)
        print()
