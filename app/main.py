"""FastAPI 应用：状态页 API（公开）+ 管理 API（token 认证）+ 调度器启动。"""
import asyncio
import json
import os
import re
import sqlite3
import time

import httpx
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from . import core, notify
from .probe import build_v1_url, engine, is_model_suspended

STATIC = Path(__file__).resolve().parent.parent / "static"
scheduler = AsyncIOScheduler()


def natural_key(s: str):
    """字母数字自然排序键：大小写不敏感，数字段按数值比较（mimo-v2 < mimo-v10）。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    core.init_db()
    # 迁移：目标已处于异常（down）但无 ongoing 事件 → 补开一条（事件跟随状态上线）。
    # 注意只补 down：partial 是瞬时抖动（如单模型一次超时），按设计不产生事件，
    # 若对 partial 补事件，重启后一轮全成功就会误发"恢复"通知（无头恢复）。
    try:
        conn = core.get_conn()
        try:
            rows = conn.execute(
                "SELECT t.id, t.status, t.last_check_at FROM targets t"
                " WHERE t.enabled=1 AND t.status='down'"
                " AND NOT EXISTS (SELECT 1 FROM incidents i"
                "  WHERE i.target_id=t.id AND i.status='ongoing')").fetchall()
            for r in rows:
                # 补事件原因：从 checks 查该目标最近一轮失败明细（模型:错误, ...）。
                # 排除 model_not_found：模型从上游 /models 消失是自动剔除逻辑，
                # 不是故障，不应写入事件原因。
                note = ""
                fr = conn.execute(
                    "SELECT group_concat(model || ':' || COALESCE(error,"
                    " 'HTTP ' || http_status), ', ') AS reason FROM ("
                    "  SELECT model, error, http_status FROM checks"
                    "  WHERE target_id=? AND layer='inference' AND model!=''"
                    "  AND ok=0 AND error != 'model_not_found'"
                    "  ORDER BY id DESC LIMIT 5)", (r["id"],)).fetchone()
                if fr and fr["reason"]:
                    note = fr["reason"]
                conn.execute(
                    "INSERT INTO incidents(target_id,started_at,status,severity,note,created_at)"
                    " VALUES(?,?,?,?,?,?)",
                    (r["id"], r["last_check_at"] or core.now_iso(),
                     "ongoing", r["status"], note, core.now_iso()))
            if rows:
                print(f"[migrate] backfilled {len(rows)} ongoing incident(s)")
            conn.commit()
        finally:
            conn.close()
    except Exception as e:
        print(f"[migrate] failed: {e}")
    if not scheduler.get_job("probe_tick"):
        scheduler.add_job(engine.tick, IntervalTrigger(seconds=5),
                          id="probe_tick", max_instances=1, coalesce=True)
    scheduler.start()
    yield
    scheduler.shutdown(wait=False)


app = FastAPI(title="AI Model Status", lifespan=lifespan)


# ---------- 认证 ----------

def require_admin(authorization: str = Header(default="")) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "unauthorized")
    user = core.verify_token(authorization[7:])
    if not user:
        raise HTTPException(401, "unauthorized or expired")
    return user


class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def api_login(body: LoginIn):
    stored_user = core.get_setting("admin_username")
    stored_hash = core.get_setting("admin_password_hash")
    if not stored_user or not stored_hash:
        raise HTTPException(500, "admin not initialized")
    if body.username != stored_user or not core.verify_password(body.password, stored_hash):
        raise HTTPException(401, "invalid credentials")
    return {"token": core.issue_token(body.username)}


# ---------- 公开状态页 API ----------

WINDOW_SECONDS = 4 * 3600      # 兜底时间轴窗口（实际窗口 = 120 × 探测间隔，见 _model_stats）
SLOW_MS = 5000                  # 首字偏慢阈值（>5s 标黄）
BARS_MAX = 120                  # 色块条固定格数：一格 = 一次探测，窗口随间隔自适应


def _model_stats(conn: sqlite3.Connection, target_id: int, model: str,
                 interval_seconds: int, suspend_fails: int = 5,
                 timeout_seconds: int = 15, retry_seconds: int = 1800) -> dict:
    """单个模型级统计（llmprobe 语义）：窗口可用率、色块条、最近状态。

    色块条固定 BARS_MAX 格，每格 = 一次探测（bar_sec = 目标配置的实际间隔），
    窗口 = bar_sec × BARS_MAX 随间隔自适应（间隔 120s → 近 4h；间隔 3000s → 近 100h）。
    """
    bar_sec = interval_seconds
    n_bars = BARS_MAX
    window_sec = bar_sec * n_bars

    # 窗口内探测记录（含延迟用于"慢"判定），按时间升序（取最近 N 条算最新可用率）
    # 注：probe.py 已按绝对时钟网格调度（间隔 = 配置值 ± 5s tick 粒度），探测开始
    # 时刻（start_at）锚定在网格上；分桶用 start_at（旧记录为 NULL 时回退 checked_at），
    # 与分桶网格同源对齐，探测耗时波动不再造成相位漂移/周期性空桶。
    rows = conn.execute(
        "SELECT checked_at, COALESCE(start_at, checked_at) AS probe_ts,"
        " ok, latency_ms FROM checks"
        " WHERE target_id=? AND layer='inference' AND model=?"
        " AND checked_at >= ? ORDER BY checked_at ASC",
        (target_id, model,
         time.strftime("%Y-%m-%d %H:%M:%S",
                       time.gmtime(time.time() - window_sec)))).fetchall()

    out = {"uptime": None, "ok_count": 0, "total_count": 0, "bars": [],
           "bar_seconds": bar_sec, "window_seconds": window_sec,
           "timeout_seconds": timeout_seconds,
           "retry_seconds": retry_seconds,
           "status": "unknown", "last_latency_ms": None, "last_ttft_ms": None,
           "last_check_at": None, "last_error": None, "last_error_at": None,
           "suspended": is_model_suspended(conn, target_id, model, suspend_fails)}
    # 最新可用率：最近 RECENT_N 次探测成功率（近期趋势，与窗口综合 uptime 互补）
    RECENT_N = 10
    recent = rows[-RECENT_N:] if len(rows) > RECENT_N else rows
    out["latest_uptime"] = round(
        100.0 * sum(r["ok"] for r in recent) / len(recent), 2) if recent else None

    if rows:
        ok_n = sum(r["ok"] for r in rows)
        out["ok_count"] = ok_n
        out["total_count"] = len(rows)
        out["uptime"] = round(100.0 * ok_n / len(rows), 2)

    # 色块条：每格 = bar_sec，绿=全过且不慢，黄=部分失败或慢，红=全败，灰=无数据
    segs: list[list[tuple[int, int]]] = [[] for _ in range(n_bars)]
    now = time.time()
    for r in rows:
        ts = core.parse_iso(r["probe_ts"])
        idx = int((now - ts) // bar_sec)
        if 0 <= idx < n_bars:
            segs[n_bars - 1 - idx].append((r["ok"], r["latency_ms"] or 0))
    for seg in segs:
        if not seg:
            out["bars"].append(None)
        else:
            ok_n = sum(ok for ok, _ in seg)
            ratio = ok_n / len(seg)
            avg_lat = sum(lat for _, lat in seg) / len(seg)
            if ratio == 1.0 and avg_lat <= SLOW_MS:
                out["bars"].append(1.0)          # 绿
            elif ratio == 0.0:
                out["bars"].append(0.0)          # 红
            else:
                out["bars"].append(0.5)          # 黄：部分失败或偏慢

    # 最新一次探测（成败皆取）：状态 + 最近探测时间
    r = conn.execute(
        "SELECT ok, latency_ms, ttft_ms, checked_at FROM checks"
        " WHERE target_id=? AND layer='inference' AND model=?"
        " ORDER BY id DESC LIMIT 1", (target_id, model)).fetchone()
    if r:
        out["status"] = "up" if r["ok"] else "down"
        out["last_check_at"] = r["checked_at"]
        # 首字延迟：仅当最新一次探测成功才展示（失败/超时的耗时不是首字）
        if r["ok"]:
            out["last_latency_ms"] = r["latency_ms"]
            # TTFT 仅流式探测有值；非流式恒为 None（前端据此自动不展示）
            out["last_ttft_ms"] = r["ttft_ms"]
    # 最近一次失败（错误原因 + 时间）
    r = conn.execute(
        "SELECT error, detail, checked_at FROM checks"
        " WHERE target_id=? AND layer='inference' AND model=? AND ok=0"
        " ORDER BY id DESC LIMIT 1", (target_id, model)).fetchone()
    if r:
        out["last_error"] = r["error"] or r["detail"] or "failed"
        out["last_error_at"] = r["checked_at"]
    return out


def _group_targets(conn: sqlite3.Connection, include_paused: bool = False) -> dict:
    """组装 渠道（目标）→ 模型 两级数据，含渠道汇总与全局统计。

    include_paused=True（admin 状态总览）：额外包含 show_on_status=0 的目标，
    渠道带 paused 标记；停测渠道不计入 summary 统计（KPI 与公开页一致）。
    """
    # 排序：全局 sort_order 决定渠道顺序；分组顺序 = 组内最小 sort_order
    # （分组名不再影响排序，与 admin 目标列表顺序一致）
    cond = "enabled=1" if include_paused else "enabled=1 AND show_on_status=1"
    targets = conn.execute(
        f"SELECT * FROM targets WHERE {cond} ORDER BY sort_order, id").fetchall()
    maintenance_ids = {
        i["target_id"] for i in conn.execute(
            "SELECT target_id FROM incidents WHERE ended_at IS NULL"
            " AND status='maintenance'").fetchall()}

    groups: dict[str, list[dict]] = {}
    total_models = 0
    available = 0
    abnormal = 0
    for row in targets:
        t = dict(row)
        models = [r["model"] for r in conn.execute(
            "SELECT DISTINCT model FROM checks"
            " WHERE target_id=? AND layer='inference' AND model != ''"
            " ORDER BY model", (t["id"],)).fetchall()]
        explicit = [m for m in (t["model_name"] or "").split(",") if m.strip()]
        # 动态模式（无显式列表）：以最新快照（/models 权威列表）为展示基准，
        # 已移除的历史模型不显示，新增模型即使还没探测也显示（unknown 占位）
        snapshot = []
        if t["model_snapshot"]:
            try:
                snapshot = json.loads(t["model_snapshot"])
            except Exception:
                snapshot = []
        if not explicit and snapshot:
            models = list(snapshot)
        else:
            for m in explicit:
                if m not in models:
                    models.append(m)
        # 过滤已移除模型：最近一次探测返回 model_not_found → 从状态页消失
        removed = {r["model"] for r in conn.execute(
            "SELECT c1.model FROM checks c1"
            " WHERE c1.target_id=? AND c1.layer='inference' AND c1.model != ''"
            " AND c1.id = (SELECT MAX(c2.id) FROM checks c2"
            "              WHERE c2.target_id=c1.target_id"
            "                AND c2.layer='inference' AND c2.model=c1.model)"
            " AND c1.error='model_not_found'", (t["id"],)).fetchall()}
        models = [m for m in models if m not in removed]
        # 统一按字母数字自然排序展示（覆盖 ASCII/上游快照顺序，首字母相同比第二字母，数字按数值）
        models = sorted(models, key=natural_key)

        suspend_fails = t.get("suspend_fails") or int(
            core.get_setting("model_suspend_fails", "5"))
        rr_val = t.get("suspend_retry_seconds")
        retry_seconds = int(rr_val) if rr_val is not None else int(
            core.get_setting("suspend_retry_seconds", "1800"))
        model_items = [_model_stats(conn, t["id"], m, t["interval_seconds"],
                                    suspend_fails, t["timeout_seconds"],
                                    retry_seconds)
                       for m in models]
        for m, item in zip(models, model_items):
            item["name"] = m

        in_maintenance = t["id"] in maintenance_ids
        ok_n = sum(1 for i in model_items if i["status"] == "up")
        down_n = sum(1 for i in model_items if i["status"] == "down")
        # 渠道可用率：窗口内该渠道全部模型探测合并
        uptime = None
        r = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(ok),0) AS ok FROM checks"
            " WHERE target_id=? AND layer='inference' AND model != ''"
            " AND checked_at >= ?",
            (t["id"], time.strftime("%Y-%m-%d %H:%M:%S",
                                    time.gmtime(time.time() - WINDOW_SECONDS)))).fetchone()
        if r["n"]:
            uptime = round(100.0 * r["ok"] / r["n"], 2)

        if in_maintenance:
            status = "maintenance"
        elif ok_n == 0 and down_n == 0:
            status = "unknown"
        elif down_n == 0:
            status = "up"
        elif ok_n == 0:
            status = "down"
        else:
            status = "partial"
        paused = t["show_on_status"] == 0
        if not paused:
            for i in model_items:
                total_models += 1
                if i["status"] == "up":
                    available += 1
                elif i["status"] == "down" and not in_maintenance:
                    abnormal += 1

        groups.setdefault(t["group_name"] or "默认", []).append({
            "id": t["id"], "name": t["name"], "type": t["type"],
            "status": status, "uptime": uptime,
            "models_ok": ok_n, "models_total": len(model_items),
            "bar_seconds": model_items[0]["bar_seconds"] if model_items else 120,
            "timeout_seconds": t["timeout_seconds"],
            "paused": paused,
            "models": model_items,
        })

    return {"groups": groups, "total_models": total_models,
            "available": available, "abnormal": abnormal}


@app.get("/api/status")
def api_status(include_paused: bool = False):
    conn = core.get_conn()
    try:
        data = _group_targets(conn, include_paused=include_paused)
        ongoing_incidents = conn.execute(
            "SELECT i.id, i.target_id, i.started_at, i.status, i.severity, i.note,"
            " t.name AS target_name"
            " FROM incidents i JOIN targets t ON t.id=i.target_id"
            " WHERE i.ended_at IS NULL ORDER BY i.started_at DESC").fetchall()
        # 已解决事件仅保留 2 小时窗口（status 页与 admin 状态总览展示用；完整历史在故障事件 tab）
        recent_incidents = conn.execute(
            "SELECT i.*, t.name AS target_name FROM incidents i"
            " JOIN targets t ON t.id=i.target_id"
            " WHERE i.ended_at IS NOT NULL"
            " AND i.ended_at >= datetime('now','-2 hours')"
            " ORDER BY i.started_at DESC LIMIT 50").fetchall()

        # 全局 KPI
        r = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(ok),0) AS ok FROM checks"
            " WHERE layer='inference' AND model != '' AND checked_at >= ?",
            (time.strftime("%Y-%m-%d %H:%M:%S",
                           time.gmtime(time.time() - WINDOW_SECONDS)),)).fetchone()
        uptime = round(100.0 * r["ok"] / r["n"], 2) if r["n"] else None
        rounds = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE layer='inference' AND model != ''"
        ).fetchone()["n"]

        summary = {
            "available": data["available"],
            "total_models": data["total_models"],
            "uptime": uptime,
            # 最新可用率：基于各模型最近一次探测（available/total_models 的百分比，
            # 排除停用/隐藏目标），与窗口综合 uptime 并列展示
            "latest_uptime": round(100.0 * data["available"] / data["total_models"], 2)
            if data["total_models"] else None,
            "channels": sum(1 for items in data["groups"].values()
                            for i in items if not i["paused"]),
            # 故障渠道：有 ongoing 故障/部分故障事件的目标数（维护中不计）
            "abnormal_channels": len({i["target_id"] for i in ongoing_incidents
                                      if i["status"] == "ongoing"}),
            "abnormal": data["abnormal"],
            "rounds": rounds,
        }

        # 时间轴窗口 = 各目标自适应窗口的最大值（随探测间隔变化）
        # 停测渠道不计入（与公开页口径一致）
        wins = [m["window_seconds"]
                for items in data["groups"].values()
                for i in items if not i["paused"] for m in i["models"]]
        window_seconds = max(wins) if wins else WINDOW_SECONDS

        return {
            "generated_at": core.now_iso(),
            "site_title": core.get_setting("site_title", "AI Model Status"),
            "announcement": core.get_setting("announcement"),
            "window_seconds": window_seconds,
            "summary": summary,
            "groups": [{"group": g, "targets": items}
                       for g, items in data["groups"].items()],
            "ongoing_incidents": [dict(i) for i in ongoing_incidents],
            "recent_incidents": [dict(i) for i in recent_incidents],
        }
    finally:
        conn.close()


# ---------- 管理 API ----------

class TargetIn(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    group_name: str = Field(default="默认", max_length=32)
    type: str = Field(default="openai", pattern="^(openai|openai-stream|anthropic|anthropic-stream)$")
    base_url: str = Field(min_length=1, max_length=256)
    api_key: str = Field(default="", max_length=512)
    model_name: str = Field(default="", max_length=512)
    probe_mode: str = Field(default="both", pattern="^(connectivity|inference|both)$")
    interval_seconds: int = Field(default=300, le=86400)
    timeout_seconds: int = Field(default=15, ge=1, le=120)
    fail_threshold: int = Field(default=3, ge=1, le=10)
    recover_threshold: int = Field(default=3, ge=1, le=10)
    suspend_fails: int | None = Field(default=None, ge=1, le=50)
    suspend_retry_seconds: int | None = Field(default=None, ge=0, le=86400)  # 0 = 停测后每轮都探测
    enabled: bool = True
    show_on_status: bool = True
    notify_enabled: bool = True
    extra_body: str = Field(default="", max_length=2048,
                            description="附加请求体参数（JSON 对象字符串，如 {\"reasoning\": {\"context\": \"auto\"}}）")


def _validate_extra_body(raw: str) -> None:
    """extra_body 必须是合法的 JSON 对象；空串合法。"""
    if not raw.strip():
        return
    try:
        eb = json.loads(raw)
    except Exception:
        raise HTTPException(400, "extra_body 必须是合法的 JSON 字符串")
    if not isinstance(eb, dict):
        raise HTTPException(400, "extra_body 必须是 JSON 对象")


def _target_out(row: sqlite3.Row) -> dict:
    t = dict(row)
    t["api_key"] = core.mask_key(core.decrypt_api_key(t.pop("api_key_enc")))
    t["enabled"] = bool(t["enabled"])
    t["show_on_status"] = bool(t["show_on_status"])
    t["notify_enabled"] = bool(t["notify_enabled"])
    return t


@app.get("/api/admin/targets", dependencies=[Depends(require_admin)])
def admin_targets():
    conn = core.get_conn()
    try:
        rows = conn.execute(
            "SELECT *, ROW_NUMBER() OVER (ORDER BY sort_order, id) AS seq FROM targets"
            " ORDER BY sort_order, id").fetchall()
        return {"targets": [_target_out(r) for r in rows]}
    finally:
        conn.close()


@app.post("/api/admin/targets", dependencies=[Depends(require_admin)])
async def admin_create_target(body: TargetIn):
    if body.interval_seconds < 60:
        raise HTTPException(400, "探测间隔过短，必须大于等于 60 秒")
    _validate_extra_body(body.extra_body)
    conn = core.get_conn()
    try:
        now = core.now_iso()
        cur = conn.execute(
            "INSERT INTO targets(name,group_name,type,base_url,api_key_enc,model_name,"
            " probe_mode,interval_seconds,timeout_seconds,fail_threshold,recover_threshold,"
            " suspend_fails,suspend_retry_seconds,"
            " enabled,show_on_status,notify_enabled,extra_body,status,created_at,updated_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (body.name, body.group_name, body.type, body.base_url,
             core.encrypt_api_key(body.api_key), body.model_name,
             body.probe_mode, body.interval_seconds, body.timeout_seconds,
             body.fail_threshold, body.recover_threshold,
             body.suspend_fails, body.suspend_retry_seconds,
             1 if body.enabled else 0, 1 if body.show_on_status else 0,
             1 if body.notify_enabled else 0,
             body.extra_body.strip(), "unknown", now, now))
        conn.commit()
        tid = cur.lastrowid
        if tid:
            max_sort = conn.execute("SELECT COALESCE(MAX(sort_order),0) FROM targets WHERE id!=?", (tid,)).fetchone()[0]
            conn.execute("UPDATE targets SET sort_order=? WHERE id=?", (max_sort + 1, tid))
            conn.commit()
    finally:
        conn.close()
    if tid is None:
        raise HTTPException(500, "create failed")
    # 创建后立即探测一轮（后台运行）
    asyncio.create_task(engine.trigger(tid))
    return {"id": tid, "pending_probe": True}


@app.put("/api/admin/targets/{tid}", dependencies=[Depends(require_admin)])
async def admin_update_target(tid: int, body: TargetIn):
    if body.interval_seconds < 60:
        raise HTTPException(400, "探测间隔过短，必须大于等于 60 秒")
    _validate_extra_body(body.extra_body)
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT * FROM targets WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise HTTPException(404, "target not found")
        enc = row["api_key_enc"]
        if body.api_key and "..." not in body.api_key:  # 空串或掩码（sk-...x9f2）= 保留原 key
            enc = core.encrypt_api_key(body.api_key)
        conn.execute(
            "UPDATE targets SET name=?,group_name=?,type=?,base_url=?,api_key_enc=?,model_name=?,"
            " probe_mode=?,interval_seconds=?,timeout_seconds=?,fail_threshold=?,recover_threshold=?,"
            " suspend_fails=?,suspend_retry_seconds=?,"
            " enabled=?,show_on_status=?,notify_enabled=?,extra_body=?,auto_disabled=0,updated_at=? WHERE id=?",
            (body.name, body.group_name, body.type, body.base_url, enc, body.model_name,
             body.probe_mode, body.interval_seconds, body.timeout_seconds,
             body.fail_threshold, body.recover_threshold,
             body.suspend_fails, body.suspend_retry_seconds,
             1 if body.enabled else 0, 1 if body.show_on_status else 0,
             1 if body.notify_enabled else 0,
             body.extra_body.strip(), core.now_iso(), tid))
        conn.commit()
    finally:
        conn.close()
    # 更新后立即探测一轮
    asyncio.create_task(engine.trigger(tid))
    return {"ok": True, "pending_probe": True}


@app.post("/api/admin/targets/sort", dependencies=[Depends(require_admin)])
def admin_reorder_targets(body: dict):
    """批量更新目标排序：body = {"order": [id1, id2, ...]}"""
    order = body.get("order", [])
    conn = core.get_conn()
    try:
        for i, tid in enumerate(order):
            conn.execute("UPDATE targets SET sort_order=? WHERE id=?", (i, tid))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.delete("/api/admin/targets/{tid}", dependencies=[Depends(require_admin)])
def admin_delete_target(tid: int):
    conn = core.get_conn()
    try:
        cur = conn.execute("DELETE FROM targets WHERE id=?", (tid,))
        conn.commit()
        if cur.rowcount == 0:
            raise HTTPException(404, "target not found")
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/admin/targets/{tid}/toggle", dependencies=[Depends(require_admin)])
def admin_toggle_target(tid: int):
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT enabled FROM targets WHERE id=?", (tid,)).fetchone()
        if row is None:
            raise HTTPException(404, "target not found")
        new_enabled = 0 if row["enabled"] else 1
        # 手动切换视为用户接管：清除自动禁用标记（不再自动恢复）
        conn.execute("UPDATE targets SET enabled=?, auto_disabled=0, updated_at=? WHERE id=?",
                     (new_enabled, core.now_iso(), tid))
        conn.commit()
        return {"enabled": bool(new_enabled)}
    finally:
        conn.close()


@app.post("/api/admin/targets/{tid}/list-models", dependencies=[Depends(require_admin)])
async def admin_list_models(tid: int):
    """实时探测上游 /models，返回当前模型列表（管理页手动触发）。"""
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT * FROM targets WHERE id=?",
                           (tid,)).fetchone()
        if row is None:
            raise HTTPException(404, "target not found")
        t = dict(row)
        t["api_key"] = core.decrypt_api_key(t["api_key_enc"])
    finally:
        conn.close()
    timeout = max(1, t["timeout_seconds"])
    async with httpx.AsyncClient(timeout=timeout) as client:
        url = build_v1_url(t["base_url"], "/models")
        headers = {"authorization": f"Bearer {t['api_key']}"} \
            if t.get("api_key") else {}
        try:
            r = await client.get(url, headers=headers)
        except Exception as e:
            return {"ok": False, "status_code": None,
                    "error": str(e)[:400], "models": []}
        if r.status_code != 200:
            return {"ok": False, "status_code": r.status_code,
                    "error": r.text[:400], "models": []}
        try:
            data = r.json()
            models = [m["id"] for m in data.get("data", [])
                      if isinstance(m, dict) and m.get("id")]
            return {"ok": True, "count": len(models), "models": models}
        except Exception as e:
            return {"ok": False, "status_code": r.status_code,
                    "error": f"解析失败: {e}", "models": []}


@app.post("/api/admin/targets/{tid}/test", dependencies=[Depends(require_admin)])
async def admin_test_target(tid: int):
    try:
        results = await engine.test_target(tid)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"results": results}


@app.get("/api/admin/checks", dependencies=[Depends(require_admin)])
def admin_checks(target_id: int = Query(ge=1), page: int = Query(default=1, ge=1),
                 page_size: int = Query(default=50, ge=1, le=200)):
    conn = core.get_conn()
    try:
        total = conn.execute(
            "SELECT COUNT(*) AS n FROM checks WHERE target_id=?", (target_id,)).fetchone()["n"]
        rows = conn.execute(
            "SELECT id,target_id,layer,model,ok,latency_ms,ttft_ms,http_status,error,detail,checked_at"
            " FROM checks WHERE target_id=? ORDER BY id DESC LIMIT ? OFFSET ?",
            (target_id, page_size, (page - 1) * page_size)).fetchall()
        return {"total": total, "page": page, "page_size": page_size,
                "checks": [dict(r) for r in rows]}
    finally:
        conn.close()


class IncidentIn(BaseModel):
    note: str = Field(default="", max_length=500)
    # ongoing=手动故障记录（前端"故障记录"选项）；resolved/maintenance 用于 PUT 更新
    status: str | None = Field(default=None, pattern="^(resolved|maintenance|ongoing)$")
    target_id: int | None = None


@app.get("/api/admin/incidents", dependencies=[Depends(require_admin)])
def admin_incidents(open_only: bool = False):
    conn = core.get_conn()
    try:
        if open_only:
            rows = conn.execute(
                "SELECT i.*, t.name AS target_name FROM incidents i"
                " JOIN targets t ON t.id=i.target_id"
                " WHERE i.ended_at IS NULL ORDER BY i.started_at DESC").fetchall()
        else:
            rows = conn.execute(
                "SELECT i.*, t.name AS target_name FROM incidents i"
                " JOIN targets t ON t.id=i.target_id"
                " ORDER BY i.started_at DESC LIMIT 200").fetchall()
        return {"incidents": [dict(r) for r in rows]}
    finally:
        conn.close()


@app.put("/api/admin/incidents/{iid}", dependencies=[Depends(require_admin)])
def admin_update_incident(iid: int, body: IncidentIn):
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT * FROM incidents WHERE id=?", (iid,)).fetchone()
        if row is None:
            raise HTTPException(404, "incident not found")
        if body.status == "resolved" and row["ended_at"] is None:
            dur = int(time.time() - core.parse_iso(row["started_at"]))
            conn.execute(
                "UPDATE incidents SET ended_at=?, duration_seconds=?, status='resolved'"
                " WHERE id=?", (core.now_iso(), dur, iid))
        elif body.status == "maintenance":
            conn.execute("UPDATE incidents SET status='maintenance' WHERE id=?", (iid,))
        if body.note is not None:
            conn.execute("UPDATE incidents SET note=? WHERE id=?", (body.note, iid))
        conn.commit()
        return {"ok": True}
    finally:
        conn.close()


@app.delete("/api/admin/incidents/{iid}", dependencies=[Depends(require_admin)])
async def admin_delete_incident(iid: int):
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT target_id FROM incidents WHERE id=?", (iid,)).fetchone()
        if row is None:
            raise HTTPException(404, "incident not found")
        conn.execute("DELETE FROM incidents WHERE id=?", (iid,))
        conn.commit()
        # 手动事件删除 → TG 立即取消置顶并删除对应消息（后台任务）
        try:
            asyncio.create_task(notify.unpin_for_target(row["target_id"]))
        except Exception:
            pass
        return {"ok": True}
    finally:
        conn.close()


@app.post("/api/admin/incidents", dependencies=[Depends(require_admin)])
async def admin_create_incident(body: IncidentIn):
    """手动创建事件（维护 / 故障记录），创建后立即 TG 通知（区别于自动事件通知）。"""
    if not body.target_id:
        raise HTTPException(400, "target_id required")
    conn = core.get_conn()
    try:
        row = conn.execute("SELECT 1 FROM targets WHERE id=?", (body.target_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "target not found")
        now = core.now_iso()
        cur = conn.execute(
            "INSERT INTO incidents(target_id,started_at,status,note,created_at)"
            " VALUES(?,?,?,?,?)",
            (body.target_id, now, body.status or "maintenance", body.note, now))
        conn.commit()
        # 手动创建 → 立即推送通知（后台任务，不阻塞响应），并置顶（TTL 2h 自动撤下）
        try:
            trow = conn.execute(
                "SELECT name FROM targets WHERE id=?", (body.target_id,)).fetchone()
            if trow:
                asyncio.create_task(
                    notify.send_manual_incident(
                        trow["name"], body.status or "maintenance",
                        body.note, target_id=body.target_id))
        except Exception:
            pass
        return {"id": cur.lastrowid}
    finally:
        conn.close()


@app.get("/api/admin/settings", dependencies=[Depends(require_admin)])
def admin_get_settings():
    return {
        "site_title": core.get_setting("site_title", "AI Model Status"),
        "announcement": core.get_setting("announcement"),
        "telegram_bot_token": core.get_setting("telegram_bot_token"),
        "telegram_chat_id": core.get_setting("telegram_chat_id"),
        "bark_url": core.get_setting("bark_url"),
        "checks_retention_days": int(core.get_setting("checks_retention_days", "7")),
        "model_suspend_fails": int(core.get_setting("model_suspend_fails", "5")),
        "model_alert_fails": int(core.get_setting("model_alert_fails", "3")),
        "suspend_retry_seconds": int(core.get_setting("suspend_retry_seconds", "1800")),
    }


class SettingsIn(BaseModel):
    site_title: str = Field(default="AI Model Status", max_length=64)
    announcement: str = Field(default="", max_length=500)
    telegram_bot_token: str = Field(default="", max_length=128)
    telegram_chat_id: str = Field(default="", max_length=64)
    bark_url: str = Field(default="", max_length=256)
    checks_retention_days: int = Field(default=7, ge=1, le=90)
    model_suspend_fails: int = Field(default=5, ge=1, le=50)
    model_alert_fails: int = Field(default=3, ge=1, le=50)
    suspend_retry_seconds: int = Field(default=1800, ge=0, le=86400)  # 0 = 停测后每轮都探测


@app.put("/api/admin/settings", dependencies=[Depends(require_admin)])
def admin_update_settings(body: SettingsIn):
    for k, v in body.model_dump().items():
        core.set_setting(k, v)
    return {"ok": True}


@app.get("/api/admin/health", dependencies=[Depends(require_admin)])
def admin_health():
    db_size = 0
    if core.DB_PATH.exists():
        db_size = core.DB_PATH.stat().st_size
    conn = core.get_conn()
    try:
        n_checks = conn.execute("SELECT COUNT(*) AS n FROM checks").fetchone()["n"]
        job = scheduler.get_job("probe_tick")
        next_run = job.next_run_time if job else None
    finally:
        conn.close()
    return {
        "scheduler_running": scheduler.running,
        "next_tick": str(next_run) if next_run else None,
        "db_size_bytes": db_size,
        "total_checks": n_checks,
        "data_dir": str(core.DATA_DIR),
    }


# ---------- 页面 ----------

@app.get("/", include_in_schema=False)
def page_index():
    return FileResponse(STATIC / "index.html")


@app.get("/admin", include_in_schema=False)
def page_admin():
    return FileResponse(STATIC / "admin.html")


# ---------- PWA 资源 ----------

@app.get("/manifest.json", include_in_schema=False)
def pwa_manifest():
    return FileResponse(STATIC / "manifest.json", media_type="application/manifest+json")


@app.get("/manifest-admin.json", include_in_schema=False)
def pwa_manifest_admin():
    return FileResponse(STATIC / "manifest-admin.json", media_type="application/manifest+json")


@app.get("/sw.js", include_in_schema=False)
def pwa_sw():
    return FileResponse(STATIC / "sw.js", media_type="application/javascript")


@app.get("/icon-512.png", include_in_schema=False)
def pwa_icon512():
    return FileResponse(STATIC / "icon-512.png", media_type="image/png")


@app.get("/icon-192.png", include_in_schema=False)
def pwa_icon192():
    return FileResponse(STATIC / "icon-192.png", media_type="image/png")


@app.get("/icon-maskable-512.png", include_in_schema=False)
def pwa_icon_maskable():
    return FileResponse(STATIC / "icon-maskable-512.png", media_type="image/png")


@app.get("/icon-180.png", include_in_schema=False)
def pwa_icon180():
    return FileResponse(STATIC / "icon-180.png", media_type="image/png")
