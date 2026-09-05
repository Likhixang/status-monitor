"""告警推送：Telegram（富文本 HTML）/ Bark。配置缺失或推送失败均静默。

置顶管理：目标故障（分组或模型级）置顶后长期挂着（无 TTL），直到目标变绿才撤下删除；
变绿通知不置顶，发出 30 秒后自动删除（RECOVERY_MSG_TTL）。
增量更新：已有故障置顶时，新模型故障不新置顶，用 editMessageText 修改原置顶消息。
"""

import asyncio
import html
import json
import sqlite3
import time

import httpx

from . import core

CONFIG_KEYS = ("telegram_bot_token", "telegram_chat_id", "bark_url")
RECOVERY_MSG_TTL = 30      # 恢复/变绿通知发出 30 秒后自动删除
SLOW_TTFT_MS = 5000        # 首字时间 > 5s 视为慢（色块条标黄口径，与 main.py 一致）


def now_cn() -> str:
    """Asia/Shanghai 时间（UTC+8，中国无夏令时）。通知展示用。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(time.time() + 8 * 3600))


def get_config() -> dict:
    conn = core.get_conn()
    try:
        out = {}
        for k in CONFIG_KEYS:
            row = conn.execute("SELECT value FROM settings WHERE key=?", (k,)).fetchone()
            out[k] = row["value"] if row else ""
        return out
    finally:
        conn.close()


def _strip_tags(s: str) -> str:
    """去掉 HTML 标签并反转义实体，供 Bark 等纯文本通道使用。"""
    return html.unescape(
        s.replace("<b>", "").replace("</b>", "").replace("<code>", "").replace("</code>", ""))


# ---------- 通知去重日志 ----------

def _log_notification(target_id: int, kind: str, model: str = "") -> None:
    """记录一条已发送的通知（用于模型级告警/恢复对称性检查）。失败静默。"""
    try:
        conn = core.get_conn()
        try:
            conn.execute(
                "INSERT INTO notification_log(target_id, kind, model, sent_at) VALUES(?,?,?,?)",
                (target_id, kind, model, core.now_iso()))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def _recent_model_alert(target_id: int, model: str) -> bool:
    """该模型最近是否发过 model_alert 故障通知（未消费 = 恢复对称性检查）。"""
    try:
        conn = core.get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM notification_log WHERE target_id=? AND kind='model_alert'"
                " AND model=? ORDER BY id DESC LIMIT 1",
                (target_id, model)).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def _model_should_alert(conn: sqlite3.Connection, target_id: int, model: str) -> bool:
    """该模型是否需要（重新）告警。

    去掉了连续失败防抖：任一模型本轮失败即报。防重复：
    最近一条记录仍是 model_alert（已告警未恢复）时不重复推送；
    恢复（model_recovered 消费）后再失败会重新告警。
    """
    last = conn.execute(
        "SELECT kind FROM notification_log WHERE target_id=? AND model=? AND"
        " kind IN ('model_alert','model_recovered') ORDER BY id DESC LIMIT 1",
        (target_id, model)).fetchone()
    return last is None or last["kind"] == "model_recovered"


def _consume_model_alert(target_id: int, model: str) -> None:
    """消费一条故障告警记录（恢复后删除，一条故障 ↔ 一条恢复）。失败静默。"""
    try:
        conn = core.get_conn()
        try:
            conn.execute(
                "DELETE FROM notification_log WHERE target_id=? AND kind='model_alert' AND model=?",
                (target_id, model))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def format_duration(seconds: int | None) -> str:
    """秒数 → 人类可读时长：2 分 34 秒 / 1 小时 5 分 / 2 天 3 小时。"""
    if seconds is None:
        return ""
    s = int(seconds)
    if s < 60:
        return f"{s} 秒"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} 分 {s} 秒" if s else f"{m} 分钟"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h} 小时 {m} 分" if m else f"{h} 小时"
    d, h = divmod(h, 24)
    return f"{d} 天 {h} 小时" if h else f"{d} 天"


def _last_incident(target_id: int, window: int = 300) -> dict | None:
    """最近一次事件（ongoing 或 resolved）的 (duration_seconds, note)。查不到返回 None。

    只认 window 秒内的事件：通知按色条即时判定（与状态机解耦），模型级抖动
    故障不产生 incident；若直接引用历史 incident，恢复通知的持续时长会永远
    显示上一次真实故障的值（陈旧引用）。真 down→up 时 incident 刚 resolve
    （ended_at≈现在），落在窗口内正常返回。
    """
    try:
        conn = core.get_conn()
        try:
            row = conn.execute(
                "SELECT status, started_at, ended_at, duration_seconds, note FROM incidents"
                " WHERE target_id=? AND status IN ('ongoing','resolved')"
                " ORDER BY id DESC LIMIT 1", (target_id,)).fetchone()
            if row is None:
                return None
            d = dict(row)
            now = time.time()
            if d["status"] == "resolved":
                # 事件已坐实关闭：只认刚结束的（ended_at 在窗口内，对应本轮恢复）
                if not d["ended_at"] or now - core.parse_iso(d["ended_at"]) > window:
                    return None
                return {"duration_seconds": d["duration_seconds"], "note": d["note"]}
            # 事件尚未坐实关闭：按已持续时长估算（只认窗口内开立的）
            if now - core.parse_iso(d["started_at"]) > window:
                return None
            return {"duration_seconds": int(now - core.parse_iso(d["started_at"])),
                    "note": d["note"]}
        finally:
            conn.close()
    except Exception:
        return None


# ---------- Telegram 底层 ----------

async def _send_telegram(token: str, chat_id: str, text: str) -> int | None:
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.post(f"https://api.telegram.org/bot{token}/sendMessage",
                         json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"})
        try:
            d = r.json()
            if d.get("ok"):
                return d["result"]["message_id"]
        except Exception:
            pass
        return None


async def _edit_message(token: str, chat_id: str, message_id: int, text: str) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"https://api.telegram.org/bot{token}/editMessageText",
                     json={"chat_id": chat_id, "message_id": message_id,
                           "text": text, "parse_mode": "HTML"})


async def _send_bark(url: str, title: str, body: str) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(url, json={"title": title, "body": body})


async def _pin_message(token: str, chat_id: str, message_id: int) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"https://api.telegram.org/bot{token}/pinChatMessage",
                     json={"chat_id": chat_id, "message_id": message_id,
                           "disable_notification": True})


async def _unpin_message(token: str, chat_id: str, message_id: int) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"https://api.telegram.org/bot{token}/unpinChatMessage",
                     json={"chat_id": chat_id, "message_id": message_id})


async def _delete_message(token: str, chat_id: str, message_id: int) -> None:
    async with httpx.AsyncClient(timeout=10) as c:
        await c.post(f"https://api.telegram.org/bot{token}/deleteMessage",
                     json={"chat_id": chat_id, "message_id": message_id})


async def _delete_after(token: str, chat_id: str, message_id: int,
                        seconds: int) -> None:
    """后台任务：seconds 秒后删除消息（失败静默：消息已被编辑/删除不报错）。"""
    await asyncio.sleep(seconds)
    try:
        await _delete_message(token, chat_id, message_id)
    except Exception:
        pass


def _load_fault_models(models_json: str) -> list[dict]:
    try:
        m = json.loads(models_json or "[]")
        return m if isinstance(m, list) else []
    except Exception:
        return []


def _fault_text(name: str, kind: str, models: list[dict],
                err_text: str | None) -> str:
    """重建置顶消息文本：kind=group → 分组故障（不列模型明细）；model → 按模型声明。"""
    now = now_cn()
    if kind == "group":
        title = f"🔴 <b>{name}</b> 分组故障"
        body = (f"━━━━━━━━━━━━━━━\n"
                f"状态: ❌ <b>不可用</b>（全部模型故障）\n"
                f"时间: <code>{now}</code>\n"
                f"错误: <code>{html.escape(err_text or '未知')}</code>")
    else:
        title = f"🟡 <b>{name}</b> 部分故障"
        lines = "\n".join(
            f"• <b>{html.escape(m['model'])}</b>: <code>{html.escape(m['error'])}</code>"
            for m in models)
        body = (f"━━━━━━━━━━━━━━━\n"
                f"数量: {len(models)} 个\n"
                f"时间: <code>{now}</code>\n"
                f"{lines}")
    return f"{title}\n{body}"


async def _set_fault_pin(target_id: int, target_name: str, kind: str,
                         models: list[dict], err_text: str | None) -> None:
    """置顶/编辑故障消息：无置顶 → 新发 + pin；有 → editMessageText 更新。失败静默。

    手动事件置顶（kind='manual'）不被自动故障覆盖。
    """
    cfg = get_config()
    token = cfg["telegram_bot_token"]
    chat_id = cfg["telegram_chat_id"]
    if not token or not chat_id:
        return
    text = _fault_text(target_name, kind, models, err_text)
    conn = core.get_conn()
    try:
        row = conn.execute(
            "SELECT chat_id, message_id, kind FROM pin_state WHERE target_id=?",
            (target_id,)).fetchone()
        if row and row["kind"] == "manual":
            return  # 手动事件置顶不被动
        if row:
            try:
                await _edit_message(token, row["chat_id"], row["message_id"], text)
            except Exception:
                pass
            conn.execute(
                "UPDATE pin_state SET kind=?, models=?, pinned_at=? WHERE target_id=?",
                (kind, json.dumps(models, ensure_ascii=False), core.now_iso(), target_id))
        else:
            mid = await _send_telegram(token, chat_id, text)
            if mid is None:
                return
            try:
                await _pin_message(token, chat_id, mid)
            except Exception:
                pass
            conn.execute(
                "INSERT OR REPLACE INTO pin_state(target_id, chat_id, message_id,"
                " pinned_at, kind, models) VALUES(?,?,?,?,?,?)",
                (target_id, int(chat_id), mid, core.now_iso(), kind,
                 json.dumps(models, ensure_ascii=False)))
        conn.commit()
    finally:
        conn.close()


async def _clear_fault_pin(target_id: int) -> None:
    """撤下并删除故障置顶（目标变绿时调用）。失败静默。"""
    cfg = get_config()
    token = cfg["telegram_bot_token"]
    if not token:
        return
    conn = core.get_conn()
    try:
        row = conn.execute(
            "SELECT chat_id, message_id FROM pin_state WHERE target_id=?",
            (target_id,)).fetchone()
        if not row:
            return
        try:
            await _unpin_message(token, row["chat_id"], row["message_id"])
        except Exception:
            pass
        try:
            await _delete_message(token, row["chat_id"], row["message_id"])
        except Exception:
            pass
        conn.execute("DELETE FROM pin_state WHERE target_id=?", (target_id,))
        conn.commit()
    finally:
        conn.close()


async def unpin_for_target(target_id: int) -> None:
    """删除事件/手动解除时调用：取消该目标当前置顶并删除消息（同步 TG）。"""
    await _clear_fault_pin(target_id)


# ---------- 通知发送 ----------

async def _send_notice(title: str, body: str, pin: bool = False) -> None:
    """通用通知：TG（pin=True 置顶；False 发完 30 秒后自动删除）+ Bark（纯文本）。"""
    cfg = get_config()
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        text = f"{title}\n{body}"
        mid = await _send_telegram(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], text)
        if mid is not None and not pin:
            asyncio.create_task(
                _delete_after(cfg["telegram_bot_token"], cfg["telegram_chat_id"],
                              mid, RECOVERY_MSG_TTL))
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))


async def send_manual_incident(target_name: str, status: str,
                               note: str = "",
                               target_id: int | None = None) -> None:
    """手动创建事件（维护/故障记录）的即时通知。

    与自动触发的故障置顶区分：管理员主动录入，立即推送一条独立格式的消息并置顶
    （pin_state 记录 kind='manual'，自动故障不覆盖）。Web 删除事件时撤下删除。
    """
    name = html.escape(target_name)
    if status == "maintenance":
        title = f"🛠️ <b>{name}</b> 手动维护"
        kind_line = "类型: 🟡 <b>维护</b>"
    else:  # ongoing 手动故障记录
        title = f"📝 <b>{name}</b> 手动故障记录"
        kind_line = "类型: 🔴 <b>故障记录</b>"
    body = (f"━━━━━━━━━━━━━━━\n"
            f"{kind_line}\n"
            f"目标: <code>{name}</code>\n"
            f"时间: <code>{now_cn()}</code>")
    if note:
        body += f"\n备注: <code>{html.escape(note)}</code>"
    cfg = get_config()
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        text = f"{title}\n{body}"
        mid = await _send_telegram(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], text)
        if mid is not None and target_id is not None:
            try:
                await _pin_message(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], mid)
                conn = core.get_conn()
                try:
                    conn.execute(
                        "INSERT OR REPLACE INTO pin_state(target_id, chat_id,"
                        " message_id, pinned_at, kind, models) VALUES(?,?,?,?,?,?)",
                        (target_id, int(cfg["telegram_chat_id"]), mid,
                         core.now_iso(), "manual", ""))
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))


async def _send_model_recovered(target_name: str,
                                models: list[tuple[str, int | None]],
                                target_id: int) -> None:
    """模型恢复通知（模型级）：不置顶，30 秒后自动删除。消费对应告警记录。"""
    if not models:
        return
    for m, _ in models:
        _consume_model_alert(target_id, m)
    name = html.escape(target_name)
    lines = "\n".join(
        f"• <b>{html.escape(m)}</b>: <code>{lat / 1000:.1f}s</code>" if lat is not None
        else f"• <b>{html.escape(m)}</b>"
        for m, lat in models)
    title = f"🟢 <b>{name}</b> 模型恢复"
    body = (f"━━━━━━━━━━━━━━━\n"
            f"数量: {len(models)} 个\n"
            f"时间: <code>{now_cn()}</code>\n"
            f"{lines}")
    await _send_notice(title, body)


async def _send_target_recovered(target_name: str, target_id: int) -> None:
    """变绿通知（目标级）：不置顶，30 秒后自动删除。附持续时长与原因。"""
    name = html.escape(target_name)
    title = f"🟢 <b>{name}</b> 恢复"
    body = (f"━━━━━━━━━━━━━━━\n"
            f"状态: ✅ <b>可用</b>\n"
            f"时间: <code>{now_cn()}</code>")
    inc = _last_incident(target_id)
    if inc:
        if inc.get("note"):
            body += f"\n原因: <code>{html.escape(str(inc['note']))}</code>"
        if inc.get("duration_seconds") is not None:
            body += f"\n持续: <code>{format_duration(inc['duration_seconds'])}</code>"
    await _send_notice(title, body)


async def sync_target_notifications(target_name: str, target_id: int,
                                    new_failed: list[tuple[str, str]],
                                    new_recovered: list[tuple[str, int | None]],
                                    all_fail: bool, all_ok: bool,
                                    err_text: str | None) -> None:
    """每轮探测后的通知同步（probe 调用；维护中 / notify_enabled=0 时调用方已跳过）。

    - 全红 → 分组故障置顶（只报分组故障，不列模型明细）；记录各模型告警供恢复对称
    - 部分红 → 模型级：新红模型记告警并入置顶；恢复模型发恢复通知（30s 删）并移出置顶
    - 全绿 → 变绿通知（30s 删）+ 撤下故障置顶
    通知与状态机事件（incidents）解耦：按本轮色条结果即时判定，不等防抖/坐实。
    """
    conn = core.get_conn()
    try:
        prow = conn.execute(
            "SELECT kind, models FROM pin_state WHERE target_id=?",
            (target_id,)).fetchone()
        cur_kind = prow["kind"] if prow else ""
        cur_models = _load_fault_models(prow["models"]) if prow else []
        cur_set = {m["model"] for m in cur_models}

        if all_fail:
            # 分组故障：记录全部失败模型告警（供恢复对称检查），置顶分组故障
            for m, _ in new_failed:
                if m is not None and _model_should_alert(conn, target_id, m):
                    _log_notification(target_id, "model_alert", m)
            if cur_kind != "group":
                await _set_fault_pin(target_id, target_name, "group", [], err_text)
            conn.commit()
            return

        # 恢复模型：仅发过故障告警的（对称性），部分红发模型级恢复、全绿合并进变绿通知
        recovered = [m for m, _ in new_recovered
                     if m is not None and _recent_model_alert(target_id, m)]
        if recovered:
            if all_ok:
                # 全绿：变绿通知覆盖，只消费告警记录
                for m in recovered:
                    _consume_model_alert(target_id, m)
            else:
                await _send_model_recovered(
                    target_name,
                    [(m, lat) for m, lat in new_recovered if m in recovered],
                    target_id)
            cur_models = [cm for cm in cur_models if cm["model"] not in recovered]

        if all_ok:
            # 变绿：此前有故障（置顶存在或刚有模型恢复）→ 变绿通知 + 撤置顶
            had_fault = bool(cur_kind) or bool(recovered)
            if had_fault and cur_kind != "manual":
                await _send_target_recovered(target_name, target_id)
                await _clear_fault_pin(target_id)
            conn.commit()
            return

        # 部分红：新失败模型 → 记告警 + 并入故障集合（更新错误信息）
        for m, err in new_failed:
            if m is None:
                continue
            if _model_should_alert(conn, target_id, m):
                _log_notification(target_id, "model_alert", m)
            if m in cur_set:
                for cm in cur_models:
                    if cm["model"] == m:
                        cm["error"] = err
                        break
            else:
                cur_models.append({"model": m, "error": err})
                cur_set.add(m)

        if cur_models:
            await _set_fault_pin(target_id, target_name, "model", cur_models, err_text)
        elif cur_kind:
            # 防御：置顶存在但无残余故障（正常应走全绿分支）→ 撤除
            await _clear_fault_pin(target_id)
        conn.commit()
    finally:
        conn.close()