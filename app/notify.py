"""告警推送：Telegram（富文本 HTML）/ Bark。配置缺失或推送失败均静默。

置顶管理：状态通知发送后 pin 到群里；同一目标状态更新时先撤旧删旧、
再置顶新的（pin_state 表记录）；超过 PIN_TTL_SECONDS 由探测 tick 自动撤下删除。
"""
import asyncio
import html
import sqlite3
import time

import httpx

from . import core

CONFIG_KEYS = ("telegram_bot_token", "telegram_chat_id", "bark_url")
PIN_TTL_SECONDS = 2 * 3600  # 状态置顶保留 2 小时后撤下删除
# 模型恢复通知发出后，该窗口内渠道整体恢复（partial/down → up）不再重复发
# 目标级恢复通知——同一恢复事件跨轮只推一条（模型先恢复、渠道后恢复的场景）
UP_SUPPRESS_WINDOW = 10 * 60
# 模型级告警防抖：连续 MODEL_ALERT_FAILS 次探测失败才推送（默认 3，与目标级一致）。
# 可通过 settings.model_alert_fails 调整，避免单次超时/抖动就发通知。
MODEL_ALERT_FAILS_DEFAULT = 3


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
    """记录一条已发送的通知（用于跨轮去重与模型级对称性检查）。失败静默。"""
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


def _recent_notification(target_id: int, kind: str, seconds: int) -> bool:
    """目标最近 seconds 秒内是否发过 kind 通知（UTC 时间比较）。"""
    try:
        conn = core.get_conn()
        try:
            row = conn.execute(
                "SELECT 1 FROM notification_log WHERE target_id=? AND kind=?"
                " AND sent_at >= datetime('now', ?) LIMIT 1",
                (target_id, kind, f"-{seconds} seconds")).fetchone()
            return row is not None
        finally:
            conn.close()
    except Exception:
        return False


def _recent_model_alert(target_id: int, model: str) -> bool:
    """该模型最近是否发过 model_alert 故障通知（用于恢复对称性检查）。"""
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


def _model_consecutive_failures(conn: sqlite3.Connection, target_id: int,
                                model: str, fails: int) -> bool:
    """模型最近 fails 次探测（含本轮）是否全部失败，且当前未处于已告警状态。

    本轮记录尚未写入 checks，故这里查的是本轮之前的最近 fails 条；
    全部失败即视为达阈值（不再要求"更早一次成功"——那会漏掉已连续
    失败超过阈值的模型，如重启/修复后错过的窗口）。
    防抖改为以 notification_log 判重：达阈值后只发一次——最近一条通知
    仍是 model_alert（已告警未恢复）时不重复推送；恢复（model_recovered）
    后再次失败会重新告警。fails<=1 退化为无防抖。
    """
    if fails <= 1:
        return True
    rows = conn.execute(
        "SELECT ok FROM checks WHERE target_id=? AND layer='inference' AND model=?"
        " ORDER BY id DESC LIMIT ?",
        (target_id, model, fails)).fetchall()
    if len(rows) < fails or not all(not r["ok"] for r in rows):
        return False
    # 防抖：已告警未恢复 → 不重复推送；恢复后再失败 → 重新告警
    last = conn.execute(
        "SELECT kind FROM notification_log WHERE target_id=? AND model=? AND"
        " kind IN ('model_alert','model_recovered') ORDER BY id DESC LIMIT 1",
        (target_id, model)).fetchone()
    return last is None or last["kind"] == "model_recovered"


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


def _recent_resolved_incident(target_id: int) -> dict | None:
    """最近一次已解决事件的 (duration_seconds, note)。查不到返回 None。"""
    try:
        conn = core.get_conn()
        try:
            row = conn.execute(
                "SELECT started_at, duration_seconds, note FROM incidents"
                " WHERE target_id=? AND status='resolved' AND ended_at IS NOT NULL"
                " ORDER BY id DESC LIMIT 1", (target_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()
    except Exception:
        return None


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


async def _replace_pin(target_id: int, chat_id: str, new_message_id: int) -> None:
    """同一目标：撤下并删除旧置顶 → 记录新置顶。失败静默（不影响通知主流程）。"""
    cfg = get_config()
    token = cfg["telegram_bot_token"]
    if not token:
        return
    conn = core.get_conn()
    try:
        row = conn.execute(
            "SELECT chat_id, message_id FROM pin_state WHERE target_id=?",
            (target_id,)).fetchone()
        if row:
            try:
                await _unpin_message(token, row["chat_id"], row["message_id"])
            except Exception:
                pass
            try:
                await _delete_message(token, row["chat_id"], row["message_id"])
            except Exception:
                pass
        conn.execute(
            "INSERT OR REPLACE INTO pin_state(target_id, chat_id, message_id, pinned_at)"
            " VALUES(?,?,?,?)",
            (target_id, int(chat_id), new_message_id, core.now_iso()))
        conn.commit()
    finally:
        conn.close()


async def cleanup_stale_pins() -> None:
    """探测 tick 调用：置顶超过 PIN_TTL_SECONDS 的撤下并删除。"""
    cfg = get_config()
    token = cfg["telegram_bot_token"]
    if not token:
        return
    cutoff = time.time() - PIN_TTL_SECONDS
    conn = core.get_conn()
    try:
        rows = conn.execute(
            "SELECT target_id, chat_id, message_id, pinned_at FROM pin_state").fetchall()
        stale = [r for r in rows if core.parse_iso(r["pinned_at"]) < cutoff]
        for r in stale:
            try:
                await _unpin_message(token, r["chat_id"], r["message_id"])
                await _delete_message(token, r["chat_id"], r["message_id"])
            except Exception:
                pass
            conn.execute("DELETE FROM pin_state WHERE target_id=?", (r["target_id"],))
        if stale:
            conn.commit()
    finally:
        conn.close()


async def unpin_for_target(target_id: int) -> None:
    """删除事件/手动解除时调用：取消该目标当前置顶并删除消息（同步 TG）。"""
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


async def send_manual_incident(target_name: str, status: str,
                               note: str = "",
                               target_id: int | None = None) -> None:
    """手动创建事件（维护/故障记录）的即时通知。

    与自动触发的 send_alert（事件开立/恢复）区分：手动创建是管理员
    主动录入，立即推送一条独立格式的消息。推送后置顶（与自动事件共用
    target_id 置顶槽位）：Web 删除事件或 2 小时 TTL 到期时自动撤下删除。
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
                await _replace_pin(target_id, cfg["telegram_chat_id"], mid)
            except Exception:
                pass
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))


async def send_alert(target_name: str, event: str, detail: str | None, target_id: int | None = None) -> None:
    name = html.escape(target_name)
    if event == "down":
        title = f"🔴 <b>{name}</b> 故障"
        body = (f"━━━━━━━━━━━━━━━\n"
                f"状态: ❌ <b>不可用</b>\n"
                f"时间: <code>{now_cn()}</code>\n"
                f"错误: <code>{html.escape(detail or '未知')}</code>")
    elif event == "up":
        title = f"🟢 <b>{name}</b> 恢复"
        body = (f"━━━━━━━━━━━━━━━\n"
                f"状态: ✅ <b>可用</b>\n"
                f"时间: <code>{now_cn()}</code>")
        # 附上本次故障的持续时长与原因（从最近 resolved 事件读取）
        if target_id is not None:
            inc = _recent_resolved_incident(target_id)
            if inc:
                if inc.get("note"):
                    body += (f"\n原因: <code>{html.escape(str(inc['note']))}</code>")
                if inc.get("duration_seconds") is not None:
                    body += (f"\n持续: <code>{format_duration(inc['duration_seconds'])}</code>")
    else:
        return

    cfg = get_config()
    # 跨轮去重：渠道恢复前刚发过模型恢复通知（同一恢复事件）→ 不再重复发
    # 目标级恢复，避免"🟢 模型恢复"+"🟢 恢复"两条。仅对真实目标生效，
    # 手动测试（target_id=None）不受抑制。
    if (event == "up" and target_id is not None
            and _recent_notification(target_id, "model_recovered", UP_SUPPRESS_WINDOW)):
        return
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        text = f"{title}\n{body}"
        mid = await _send_telegram(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], text)
        if mid is not None and target_id is not None:
            try:
                await _pin_message(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], mid)
                await _replace_pin(target_id, cfg["telegram_chat_id"], mid)
            except Exception:
                pass
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))


async def send_model_alert(target_name: str, models: list[tuple[str, str]],
                           target_id: int | None = None) -> None:
    """新增异常模型通知（模型级，与事件逻辑解耦）。

    models: [(model, error), ...] —— 同一轮同一目标合并成一条推送。
    防抖：仅当模型连续 model_alert_fails 次（默认 3）探测失败才推送，
    且该模型当前未处于已告警状态（首次达阈值才发），避免单次超时/抖动
    误报与重复轰炸。
    未配置通知渠道时静默跳过。
    """
    if not models:
        return
    fails = int(core.get_setting("model_alert_fails",
                                 str(MODEL_ALERT_FAILS_DEFAULT)))
    eligible: list[tuple[str, str]] = []
    if target_id is not None:
        conn = core.get_conn()
        try:
            for m, err in models:
                if _model_consecutive_failures(conn, target_id, m, fails):
                    eligible.append((m, err))
        finally:
            conn.close()
    else:
        eligible = list(models)
    if not eligible:
        return
    name = html.escape(target_name)
    lines = "\n".join(
        f"• <b>{html.escape(m)}</b>: <code>{html.escape(err)}</code>"
        for m, err in eligible)
    title = f"🟡 <b>{name}</b> 新增异常模型"
    body = (f"━━━━━━━━━━━━━━━\n"
            f"数量: {len(eligible)} 个\n"
            f"时间: <code>{now_cn()}</code>\n"
            f"{lines}")
    cfg = get_config()
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        text = f"{title}\n{body}"
        mid = await _send_telegram(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], text)
        if mid is not None and target_id is not None:
            # 记录每条模型的故障通知（用于恢复对称性检查）
            for m, _ in eligible:
                if m is not None:
                    _log_notification(target_id, "model_alert", m)
            try:
                await _pin_message(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], mid)
                await _replace_pin(target_id, cfg["telegram_chat_id"], mid)
            except Exception:
                pass
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))


async def send_model_recovered(target_name: str, models: list[tuple[str, int | None]],
                               target_id: int | None = None) -> None:
    """模型恢复通知（模型级，与事件逻辑解耦）。对称于 send_model_alert。

    对称性：只发发过故障通知（model_alert）的模型恢复——没发过故障的不推恢复，
    避免抖动恢复的无头推送。同一轮合并成一条推送。
    """
    if not models:
        return
    # 对称性检查：只保留发过 model_alert 的模型
    eligible: list[tuple[str, int | None]] = []
    if target_id is not None:
        for m, lat in models:
            if m is not None and _recent_model_alert(target_id, m):
                eligible.append((m, lat))
    else:
        eligible = list(models)
    if not eligible:
        return
    name = html.escape(target_name)
    lines = "\n".join(
        f"• <b>{html.escape(m)}</b>: <code>{lat / 1000:.1f}s</code>" if lat is not None
        else f"• <b>{html.escape(m)}</b>"
        for m, lat in eligible)
    title = f"🟢 <b>{name}</b> 模型恢复"
    body = (f"━━━━━━━━━━━━━━━\n"
            f"数量: {len(eligible)} 个\n"
            f"时间: <code>{now_cn()}</code>\n"
            f"{lines}")
    cfg = get_config()
    if cfg["telegram_bot_token"] and cfg["telegram_chat_id"]:
        text = f"{title}\n{body}"
        mid = await _send_telegram(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], text)
        if mid is not None and target_id is not None:
            # 发送成功即记录，供 send_alert("up") 跨轮去重
            _log_notification(target_id, "model_recovered")
            # 消费对应故障记录：一条故障 ↔ 一条恢复，避免旧记录残留导致下次抖动误发
            try:
                conn = core.get_conn()
                try:
                    for m, _ in eligible:
                        conn.execute(
                            "DELETE FROM notification_log WHERE target_id=? AND kind='model_alert' AND model=?",
                            (target_id, m))
                    conn.commit()
                finally:
                    conn.close()
            except Exception:
                pass
            try:
                await _pin_message(cfg["telegram_bot_token"],
                                   cfg["telegram_chat_id"], mid)
                await _replace_pin(target_id, cfg["telegram_chat_id"], mid)
            except Exception:
                pass
    if cfg["bark_url"]:
        await _send_bark(cfg["bark_url"], _strip_tags(title), _strip_tags(body))
