#!/usr/bin/env python3
"""
s15: 多智能体团队 (Agent Teams) — 消息总线 + 线程化队友 + 收件箱注入。

运行: python s15_agent_teams/code.py
前置要求: pip install anthropic python-dotenv 并在 .env 中配置 ANTHROPIC_API_KEY

相比 s14 的核心变动:
  - MessageBus (消息总线) 类: 基于文件的邮箱机制 (.mailboxes/*.jsonl)
  - spawn_teammate_thread: 在后台线程中孵化（创建）队友 Agent
  - Teammate (队友) 运行着属于自己的精简版 agent_loop (仅包含 bash, read, write, send_message)
  - 主节点 (Lead) 新增工具: spawn_teammate, send_message, check_inbox
  - 主节点收件箱: 队友发来的消息会被注入到历史上下文中 (不仅仅是打印)
  - 教学限制: 为了防止死循环，队友最多运行 10 轮交互 (真正的 Claude Code 会使用 idle loop)

交互流示意:
  主节点: cron_queue → messages → prompt → LLM → TOOLS ────→ loop
                  ↑                     ↓                        |
                  └── inbox ← MessageBus ← teammate.send_message ←┘
  队友: inbox → LLM → bash/read/write/send → loop (最多 10 轮)
"""

import os, subprocess, json, time, random, threading
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict

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

# ── 任务系统 (同步自 s12) ──

TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: Optional[str]
    blockedBy: list[str]


def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject, description=description,
        status="pending", owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task


def save_task(task: Task):
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))


def load_task(task_id: str) -> Task:
    return Task(**json.loads(_task_path(task_id).read_text()))


def list_tasks() -> list[Task]:
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]


def get_task(task_id: str) -> str:
    """返回 JSON 格式的完整任务详情。"""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)


def can_start(task_id: str) -> bool:
    """检查是否所有 blockedBy 的前置依赖都已完成。
    缺失的依赖同样会被视为阻塞状态。"""
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id: str, owner: str = "agent") -> str:
    task = load_task(task_id)
    if task.status != "pending":
        return f"任务 {task_id} 当前状态为 {task.status}，无法认领"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"被以下依赖阻塞: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    print(f"  \033[36m[认领任务] {task.subject} → in_progress (归属: {owner})\033[0m")
    return f"已认领 {task.id} ({task.subject})"


def complete_task(task_id: str) -> str:
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"任务 {task_id} 当前状态为 {task.status}，无法完成"
    task.status = "completed"
    save_task(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    print(f"  \033[32m[完成任务] {task.subject} ✓\033[0m")
    msg = f"已完成 {task.id} ({task.subject})"
    if unblocked:
        msg += f"\n解除阻塞: {', '.join(unblocked)}"
        print(f"  \033[33m[解除阻塞] {', '.join(unblocked)}\033[0m")
    return msg


# ── 提示词组装 (同步自 s10) ──

PROMPT_SECTIONS = {
    "identity": "你是一个代码智能体。请直接行动，不要过多解释。",
    "tools": "可用工具: bash, read_file, write_file, "
             "get_task, create_task, list_tasks, claim_task, complete_task, "
             "schedule_cron, list_crons, cancel_cron, "
             "spawn_teammate, send_message, check_inbox.",
    "workspace": f"当前工作目录: {WORKDIR}",
    "memory": "相关的历史记忆会在可用时被注入到下方。",
}


def assemble_system_prompt(context: dict) -> str:
    sections = [PROMPT_SECTIONS["identity"],
                PROMPT_SECTIONS["tools"],
                PROMPT_SECTIONS["workspace"]]
    memories = context.get("memories", "")
    if memories:
        sections.append(f"相关记忆:\n{memories}")
    return "\n\n".join(sections)


_last_context_key, _last_prompt = None, None


def get_system_prompt(context: dict) -> str:
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    return _last_prompt


# ── 基础工具 (Tools) ──

def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"路径越界，跳出了工作目录: {p}")
    return path


def run_bash(command: str, run_in_background: bool = False) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=WORKDIR,
                           capture_output=True, text=True, timeout=120)
        out = (r.stdout + r.stderr).strip()
        return out[:50000] if out else "(无输出)"
    except subprocess.TimeoutExpired:
        return "错误: 命令执行超时 (120秒)"


def run_read(path: str, limit: Optional[int] = None) -> str:
    try:
        lines = safe_path(path).read_text().splitlines()
        if limit and limit < len(lines):
            lines = lines[:limit] + [f"... (省略了剩余 {len(lines) - limit} 行)"]
        return "\n".join(lines)
    except Exception as e:
        return f"读取错误: {e}"


def run_write(path: str, content: str) -> str:
    try:
        fp = safe_path(path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
        return f"成功写入 {len(content)} 字节到 {path}"
    except Exception as e:
        return f"写入错误: {e}"


# 任务相关工具 (Task tools)

def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    task = create_task(subject, description, blockedBy)
    deps = f" (被阻塞于: {', '.join(blockedBy)})" if blockedBy else ""
    print(f"  \033[34m[创建任务] {task.subject}{deps}\033[0m")
    return f"已创建 {task.id}: {task.subject}{deps}"


def run_list_tasks() -> str:
    tasks = list_tasks()
    if not tasks:
        return "当前无任务。请使用 create_task 创建任务。"
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (被阻塞于: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)


def run_get_task(task_id: str) -> str:
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"错误: 未找到任务 {task_id}"


def run_claim_task(task_id: str) -> str:
    return claim_task(task_id, owner="agent")


def run_complete_task(task_id: str) -> str:
    return complete_task(task_id)


# ── 后台任务系统 (同步自 s13) ──

_bg_counter = 0
background_tasks: dict[str, dict] = {}
background_results: dict[str, str] = {}
background_lock = threading.Lock()


def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)


def should_run_background(tool_name: str, tool_input: dict) -> bool:
    if tool_input.get("run_in_background"):
        return True
    return is_slow_operation(tool_name, tool_input)


def execute_tool(block) -> str:
    handler = {
        "bash": run_bash, "read_file": run_read, "write_file": run_write,
        "create_task": run_create_task, "list_tasks": run_list_tasks,
        "get_task": run_get_task, "claim_task": run_claim_task,
        "complete_task": run_complete_task,
        "schedule_cron": run_schedule_cron, "list_crons": run_list_crons,
        "cancel_cron": run_cancel_cron,
        "spawn_teammate": run_spawn_teammate,
        "send_message": run_send_message, "check_inbox": run_check_inbox,
    }.get(block.name)
    if handler:
        return handler(**block.input)
    return f"未知工具: {block.name}"


def start_background_task(block) -> str:
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.input.get("command", block.name)

    def worker():
        result = execute_tool(block)
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result

    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.id,
            "command": cmd,
            "status": "running",
        }
    threading.Thread(target=worker, daemon=True).start()
    print(f"  \033[33m[后台运行] 已派发 {bg_id}: {cmd[:40]}\033[0m")
    return bg_id


def collect_background_results() -> list[str]:
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        print(f"  \033[32m[后台完成] {bg_id}: "
              f"{task['command'][:40]} (共 {len(output)} 个字符)\033[0m")
    return notifications


# ── 定时任务调度器 (同步自 s14) ──

DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"


@dataclass
class CronJob:
    id: str
    cron: str        # "0 9 * * *"
    prompt: str      # 触发时注入的提示信息
    recurring: bool  # True = 循环执行, False = 单次执行
    durable: bool    # True = 持久化到磁盘


scheduled_jobs: dict[str, CronJob] = {}
cron_queue: list[CronJob] = []
cron_lock = threading.Lock()
_last_fired: dict[str, str] = {}  # job_id → "YYYY-MM-DD HH:MM"


def _cron_field_matches(field: str, value: int) -> bool:
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7

    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)

    if not (m and h and month_ok):
        return False
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> Optional[str]:
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"步长无效: {field}"
        step = int(step_str)
        if step <= 0:
            return f"步长必须大于 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err: return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"范围无效: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"范围 {field} 越界 [{lo}-{hi}]"
        if a > b:
            return f"范围起点大于终点: {field}"
        return None
    if not field.isdigit():
        return f"无效字段: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"值 {val} 越界 [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> Optional[str]:
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"期望 5 个字段，但得到了 {len(fields)} 个"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    if not DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            err = validate_cron(job.cron)
            if err:
                print(f"  \033[31m[定时任务] 跳过无效任务 {job.id}: {err}\033[0m")
                continue
            scheduled_jobs[job.id] = job
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            print(f"  \033[35m[定时任务] 成功加载 {len(valid)} 个持久化任务\033[0m")
    except Exception:
        pass


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    print(f"  \033[35m[注册定时任务] {job.id} '{cron}' → {prompt[:40]}\033[0m")
    return job


def cancel_job(job_id: str) -> str:
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"未找到任务 {job_id}"
    if job.durable:
        save_durable_jobs()
    print(f"  \033[31m[取消定时任务] {job_id}\033[0m")
    return f"已取消 {job_id}"


def cron_scheduler_loop():
    while True:
        time.sleep(1)
        now = datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            print(f"  \033[35m[触发定时任务] {job.id} → "
                                  f"{job.prompt[:40]}\033[0m")
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    print(f"  \033[31m[定时任务错误] {job.id}: {e}\033[0m")


def consume_cron_queue() -> list[CronJob]:
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


# 启动时加载持久化任务，并开启调度器线程
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
print("  \033[35m[定时任务] 调度器线程已启动\033[0m")


# ── 定时任务工具 (Cron Tools) ──

def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"错误: {result}"
    return f"已调度 {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "当前无定时任务。请使用 schedule_cron 添加。"
    lines = []
    for j in jobs:
        tag = "循环" if j.recurring else "单次"
        dur = "持久化" if j.durable else "会话级"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    return cancel_job(job_id)


# ── 消息总线 (MessageBus, s15 新增) ──
# 教学版使用简单的文件追加写入 + 删除。
# 在真正的 Claude Code 中会使用 proper-lockfile 保证并发写入安全。

MAILBOX_DIR = WORKDIR / ".mailboxes"
MAILBOX_DIR.mkdir(exist_ok=True)


class MessageBus:
    """基于文件的消息总线。每个 Agent 都有自己的 .jsonl 邮箱。
    读取是破坏性的：读取后即刻删除文件 (消费消息)。"""

    def send(self, from_agent: str, to_agent: str, content: str,
             msg_type: str = "message"):
        msg = {"from": from_agent, "to": to_agent,
               "content": content, "type": msg_type,
               "ts": time.time()}
        inbox = MAILBOX_DIR / f"{to_agent}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(msg) + "\n")
        print(f"  \033[33m[消息总线] {from_agent} → {to_agent}: "
              f"{content[:50]}\033[0m")

    def read_inbox(self, agent: str) -> list[dict]:
        inbox = MAILBOX_DIR / f"{agent}.jsonl"
        if not inbox.exists():
            return []
        msgs = [json.loads(line) for line in inbox.read_text().splitlines()
                if line.strip()]
        inbox.unlink()  # 消费机制: 读完即删
        return msgs


BUS = MessageBus()

# 追踪当前活跃的队友 Agent
active_teammates: dict[str, bool] = {}


# ── 队友线程 (Teammate Thread, s15 新增) ──

def spawn_teammate_thread(name: str, role: str, prompt: str) -> str:
    """在后台线程中孵化队友 Agent。
    教学版限制：每个队友最多运行 10 轮。
    真正的 Claude Code：队友会使用闲置循环（等待信箱消息、执行任务、重复）直至收到关闭指令。"""
    if name in active_teammates:
        return f"队友 '{name}' 已经存在"

    system = (f"你是 '{name}'，担任 {role} 角色。"
              f"请使用工具完成任务。"
              f"通过 send_message 将结果发送给 'lead' (主节点)。")

    def run():
        messages = [{"role": "user", "content": prompt}]
        # 队友的精简版工具集
        sub_tools = [
            {"name": "bash", "description": "运行 Shell 命令。",
             "input_schema": {"type": "object",
                              "properties": {"command": {"type": "string"}},
                              "required": ["command"]}},
            {"name": "read_file", "description": "读取文件内容。",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"}},
                              "required": ["path"]}},
            {"name": "write_file", "description": "将内容写入文件。",
             "input_schema": {"type": "object",
                              "properties": {"path": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["path", "content"]}},
            {"name": "send_message",
             "description": "给另一个 Agent 发送消息。",
             "input_schema": {"type": "object",
                              "properties": {"to": {"type": "string"},
                                             "content": {"type": "string"}},
                              "required": ["to", "content"]}},
        ]
        sub_handlers = {
            "bash": run_bash, "read_file": run_read, "write_file": run_write,
            "send_message": lambda to, content: (BUS.send(name, to, content),
                                                  "已发送")[1],
        }

        # 限定最多交互 10 轮以防死循环
        for _ in range(10):
            inbox = BUS.read_inbox(name)
            if inbox:
                messages.append({"role": "user",
                                 "content": f"<inbox>{json.dumps(inbox)}</inbox>"})
            try:
                response = client.messages.create(
                    model=MODEL, system=system, messages=messages[-20:],
                    tools=sub_tools, max_tokens=8000)
            except Exception:
                break
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                break
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    handler = sub_handlers.get(block.name)
                    output = handler(**block.input) if handler else "未知命令"
                    results.append({"type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": str(output)})
            messages.append({"role": "user", "content": results})

        # 结束后向主节点 (Lead) 发送最终总结
        summary = "完成。"
        for msg in reversed(messages):
            if msg["role"] == "assistant" and isinstance(msg["content"], list):
                for b in msg["content"]:
                    if getattr(b, "type", None) == "text":
                        summary = b.text
                        break
                else:
                    continue
                break
        BUS.send(name, "lead", summary, "result")
        active_teammates.pop(name, None)
        print(f"  \033[32m[队友] {name} 执行完毕\033[0m")

    active_teammates[name] = True
    threading.Thread(target=run, daemon=True).start()
    print(f"  \033[36m[队友] 成功孵化 {name} 作为 {role}\033[0m")
    return f"已成功孵化队友 '{name}' 作为 {role}"


# ── 团队协作工具 (Team Tool Handlers, s15 新增) ──

def run_spawn_teammate(name: str, role: str, prompt: str) -> str:
    return spawn_teammate_thread(name, role, prompt)


def run_send_message(to: str, content: str) -> str:
    BUS.send("lead", to, content)
    return f"已发送至 {to}"


def run_check_inbox() -> str:
    msgs = BUS.read_inbox("lead")
    if not msgs:
        return "(收件箱为空)"
    lines = []
    for m in msgs:
        lines.append(f"  [{m['from']}] {m['content'][:200]}")
    return "\n".join(lines)


# ── 工具定义 ──

TOOLS = [
    {"name": "bash", "description": "运行 Shell 命令。",
     "input_schema": {"type": "object",
                      "properties": {
                          "command": {"type": "string"},
                          "run_in_background": {"type": "boolean"}},
                      "required": ["command"]}},
    {"name": "read_file", "description": "读取文件内容。",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "limit": {"type": "integer"}},
                      "required": ["path"]}},
    {"name": "write_file", "description": "将内容写入文件。",
     "input_schema": {"type": "object",
                      "properties": {"path": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["path", "content"]}},
    {"name": "create_task",
     "description": "创建一个新任务，可附带 blockedBy 前置依赖。",
     "input_schema": {"type": "object",
                      "properties": {
                          "subject": {"type": "string"},
                          "description": {"type": "string"},
                          "blockedBy": {"type": "array",
                                        "items": {"type": "string"}}},
                      "required": ["subject"]}},
    {"name": "list_tasks",
     "description": "列出所有任务，包含其状态、归属及依赖关系。",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "get_task",
     "description": "根据 ID 获取指定任务的完整详情。",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "claim_task",
     "description": "认领一个处于 pending 状态的任务。分配 owner 并将状态转为 in_progress。",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "complete_task",
     "description": "完成一个 in_progress 的任务。会汇报因此解除阻塞的下游任务。",
     "input_schema": {"type": "object",
                      "properties": {"task_id": {"type": "string"}},
                      "required": ["task_id"]}},
    {"name": "schedule_cron",
     "description": "调度一个定时任务。cron 表达式需 5 位: 分钟 小时 日 月 星期。",
     "input_schema": {"type": "object",
                      "properties": {
                          "cron": {"type": "string",
                                   "description": "5 位 cron 表达式"},
                          "prompt": {"type": "string",
                                     "description": "触发时注入的信息"},
                          "recurring": {"type": "boolean",
                                        "description": "True=循环执行, False=单次执行"},
                          "durable": {"type": "boolean",
                                      "description": "True=持久化到磁盘"}},
                      "required": ["cron", "prompt"]}},
    {"name": "list_crons",
     "description": "列出所有已注册的定时任务。",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
    {"name": "cancel_cron",
     "description": "根据 ID 取消一个定时任务。",
     "input_schema": {"type": "object",
                      "properties": {"job_id": {"type": "string"}},
                      "required": ["job_id"]}},
    {"name": "spawn_teammate",
     "description": "在后台线程孵化（创建）一个队友 Agent。",
     "input_schema": {"type": "object",
                      "properties": {
                          "name": {"type": "string"},
                          "role": {"type": "string"},
                          "prompt": {"type": "string"}},
                      "required": ["name", "role", "prompt"]}},
    {"name": "send_message",
     "description": "通过消息总线给队友发送消息。",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string"},
                                     "content": {"type": "string"}},
                      "required": ["to", "content"]}},
    {"name": "check_inbox",
     "description": "检查主节点 (Lead) 收件箱中是否有队友发来的消息。",
     "input_schema": {"type": "object", "properties": {},
                      "required": []}},
]


# ── 上下文环境 (Context) ──

def update_context(context: dict, messages: list) -> dict:
    """根据真实状态推导并更新上下文环境。"""
    memories = ""
    if MEMORY_INDEX.exists():
        content = MEMORY_INDEX.read_text().strip()
        if content:
            memories = content
    return {
        "enabled_tools": [t["name"] for t in TOOLS],
        "workspace": str(WORKDIR),
        "memories": memories,
    }


# ── Agent 循环 (Agent Loop) ──
# 教学版保持了基础的 Agent 循环。
# 定时任务队列在调用 agent_loop 时才会被消费；真正的 Claude Code 中会有队列处理器自动唤醒。

def agent_loop(messages: list, context: dict):
    system = get_system_prompt(context)
    while True:
        # 消费被触发的定时任务 → 将其作为 user 消息注入
        fired = consume_cron_queue()
        for job in fired:
            messages.append({"role": "user",
                             "content": f"[定时任务触发] {job.prompt}"})
            print(f"  \033[35m[注入定时任务] {job.prompt[:50]}\033[0m")

        try:
            response = client.messages.create(
                model=MODEL, system=system, messages=messages,
                tools=TOOLS, max_tokens=8000)
        except Exception as e:
            messages.append({"role": "assistant", "content": [
                {"type": "text",
                 "text": f"[错误] {type(e).__name__}: {e}"}]})
            return

        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            print(f"\033[36m> {block.name}\033[0m")

            if should_run_background(block.name, block.input):
                bg_id = start_background_task(block)
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"[后台任务 {bg_id} 已启动] "
                                           f"任务完成后将通过通知返回结果。"})
            else:
                output = execute_tool(block)
                print(str(output)[:300])
                results.append({"type": "tool_result",
                                "tool_use_id": block.id,
                                "content": output})

        # 在单次 user 消息中合并后台通知与常规工具结果
        user_content = []
        bg_notifications = collect_background_results()
        if bg_notifications:
            for notif in bg_notifications:
                user_content.append({"type": "text", "text": notif})
        user_content.extend(results)
        messages.append({"role": "user", "content": user_content})
        context = update_context(context, messages)
        system = get_system_prompt(context)


if __name__ == "__main__":
    print("s15: 多智能体团队 (agent teams)")
    print("输入问题后按回车发送。输入 q 退出。\n")
    history = []
    context = update_context({}, [])
    while True:
        try:
            query = input("\033[36ms15 >> \033[0m")
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

        # 检查收件箱中的队友消息 → 并将其注入到历史记录中
        inbox = BUS.read_inbox("lead")
        if inbox:
            inbox_text = "\n".join(
                f"来自 {m['from']}: {m['content'][:200]}" for m in inbox)
            history.append({"role": "user",
                            "content": f"[收件箱]\n{inbox_text}"})
            print(f"\n\033[33m[收件箱: 已注入 {len(inbox)} 条队友消息]\033[0m")
        print()
