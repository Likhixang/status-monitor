"""探测引擎：连通层 + 推理层探测、防抖状态机、到期调度。"""
import asyncio
import json
import sqlite3
import time

import httpx

from . import core, notify

ERROR_TIMEOUT = "timeout"
ERROR_NETWORK = "network"
ERROR_AUTH = "auth"
ERROR_RATE_LIMIT = "rate_limit"
ERROR_SERVER = "server_error"
ERROR_HTTP = "http_error"
ERROR_PARSE = "parse_error"
ERROR_STREAM = "stream_error"

# 动态模式（不填模型名）每轮最多探测的模型数
MAX_AUTO_MODELS = 50
# 数据清理间隔：每 6 小时检查一次过期数据
CLEANUP_INTERVAL = 6 * 3600
# 模型自动停测：连续失败达到该次数后暂停探测
MODEL_SUSPEND_FAILS = 5
# 停测模型的试探间隔：每隔这么久发一次探测，成功则自动恢复
SUSPEND_RETRY_SECONDS = 30 * 60


def is_model_suspended(conn: sqlite3.Connection, target_id: int, model: str,
                       fails: int = MODEL_SUSPEND_FAILS) -> bool:
    """模型最近 fails 次探测全部失败 → 判定为停测。"""
    rows = conn.execute(
        "SELECT ok FROM checks WHERE target_id=? AND layer='inference' AND model=?"
        " ORDER BY id DESC LIMIT ?",
        (target_id, model, fails)).fetchall()
    if len(rows) < fails:
        return False
    return all(not r["ok"] for r in rows)


def classify_error(status_code, exc) -> str | None:
    if isinstance(exc, httpx.TimeoutException):
        return ERROR_TIMEOUT
    if isinstance(exc, (httpx.ConnectError, httpx.ConnectTimeout,
                        httpx.ReadError, httpx.ProtocolError)):
        return ERROR_NETWORK
    if status_code is None:
        return ERROR_NETWORK
    if status_code in (401, 403):
        return ERROR_AUTH
    if status_code == 429:
        return ERROR_RATE_LIMIT
    if status_code >= 500:
        return ERROR_SERVER
    if status_code >= 400:
        return ERROR_HTTP
    return None


def build_v1_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + path
    return base + "/v1" + path


def _merge_extra_body(body: dict, target: dict) -> dict:
    """合并 target.extra_body（JSON 对象）到请求体；非法 JSON 或空串则原样返回。

    stream 由探测机制决定（SSE 解析依赖它），禁止被 extra_body 覆盖。
    """
    raw = (target.get("extra_body") or "").strip()
    if not raw:
        return body
    try:
        extra = json.loads(raw)
    except Exception:
        return body
    if not isinstance(extra, dict):
        return body
    out = dict(body)
    for k, v in extra.items():
        if k == "stream":
            continue
        out[k] = v
    return out


async def probe_connectivity(client: httpx.AsyncClient, target: dict, timeout: float) -> dict:
    t0 = time.monotonic()
    try:
        url = build_v1_url(target["base_url"], "/models")
        headers = {"authorization": f"Bearer {target['api_key']}"} if target.get("api_key") else {}
        r = await client.get(url, headers=headers, timeout=timeout)
        latency = int((time.monotonic() - t0) * 1000)
        err = classify_error(r.status_code, None)
        return {"layer": "connectivity", "ok": err is None, "latency_ms": latency,
                "http_status": r.status_code, "error": err, "detail": ""}
    except Exception as e:
        latency = int((time.monotonic() - t0) * 1000)
        return {"layer": "connectivity", "ok": False, "latency_ms": latency,
                "http_status": None, "error": classify_error(None, e),
                "detail": str(e)[:400]}


async def probe_inference_stream(client: httpx.AsyncClient, target: dict,
                                 timeout: float, model: str) -> dict:
    """流式推理探测：POST stream=true，逐行解析 SSE。

    判定规则：
    - OpenAI 兼容：收到带 delta 的 chunk 或 [DONE] 收尾 → 成功
    - Anthropic：收到 message_stop 收尾事件 → 成功
    - 流内 error 事件（HTTP 状态码可能已是 200）→ 失败（stream_error）
    - 断流：已开始生成（收到内容块/首 token）但未等来收尾事件 → stream_error（stream_truncated）
    - 有 SSE 数据但始终无内容 → network；无 data 事件（非 SSE）→ parse
    - 顺带量化 TTFT：从请求发出到首个内容 delta 的毫秒数（ttft_ms）
    """
    t0 = time.monotonic()
    is_anthropic = target["type"].startswith("anthropic")
    prompt = (target.get("prompt") or "").strip()
    msg_content = prompt if prompt else "ping"
    if is_anthropic:
        url = target["base_url"].rstrip("/") + "/v1/messages"
        headers = {"anthropic-version": "2023-06-01",
                   "content-type": "application/json"}
        if target.get("api_key"):
            headers["x-api-key"] = target["api_key"]
        # Anthropic 要求必填 max_tokens：有提示词时放宽至 4096，无提示词保持 64
        max_tok = 4096 if prompt else 64
        body = {"model": model, "max_tokens": max_tok,
                "messages": [{"role": "user", "content": msg_content}],
                "stream": True}
    else:
        url = build_v1_url(target["base_url"], "/chat/completions")
        headers = {"content-type": "application/json"}
        if target.get("api_key"):
            headers["authorization"] = f"Bearer {target['api_key']}"
        body = {"model": model,
                "messages": [{"role": "user", "content": msg_content}],
                "stream": True}
        # 有提示词时不带 max_tokens（不设限），无提示词带 64 限制快速探测
        if not prompt:
            body["max_tokens"] = 64
    body = _merge_extra_body(body, target)
    try:
        async with client.stream("POST", url, json=body, headers=headers,
                                 timeout=timeout) as r:
            if r.status_code >= 400:
                latency = int((time.monotonic() - t0) * 1000)
                detail = ""
                try:
                    await r.aread()
                    try:
                        d = r.json()
                        detail = str(d.get("error", {}).get("message", d)) if isinstance(d, dict) else str(d)
                    except Exception:
                        detail = (r.content or b"").decode(errors="replace")
                    detail = detail[:400]
                except Exception:
                    pass
                return {"layer": "inference", "model": model, "ok": False,
                        "latency_ms": latency, "http_status": r.status_code,
                        "error": classify_error(r.status_code, None),
                        "detail": detail}
            got_delta = False
            saw_end = False
            err = None
            parse_lines = 0
            first_token_ts = 0.0
            async for line in r.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                parse_lines += 1
                if payload == "[DONE]":
                    saw_end = True
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                # OpenAI 错误对象：{"error": {...}}
                if obj.get("error"):
                    er = obj["error"]
                    err = str(er.get("type") or er.get("message")
                              or "stream_error")[:200]
                    break
                if is_anthropic:
                    # 流内 error 事件：{"type":"error","error":{...}}
                    if obj.get("type") == "error":
                        er = obj.get("error") or {}
                        err = str(er.get("type") or "stream_error")[:200]
                        break
                    # 首个内容块到达 = TTFT 时点
                    if (obj.get("type") in ("content_block_start", "content_block_delta")
                            and not first_token_ts):
                        first_token_ts = time.monotonic()
                    if obj.get("type") == "message_stop":
                        saw_end = True
                        break
                else:
                    ch = obj.get("choices")
                    if (isinstance(ch, list) and ch
                            and isinstance(ch[0], dict) and ch[0].get("delta")):
                        # chunk 到达 = TTFT 时点（含纯 role 的起始 chunk，首个就计）
                        if not first_token_ts:
                            first_token_ts = time.monotonic()
                        got_delta = True
            latency = int((time.monotonic() - t0) * 1000)
            ttft = int((first_token_ts - t0) * 1000) if first_token_ts else None
            if err:
                return {"layer": "inference", "model": model, "ok": False,
                        "latency_ms": latency, "ttft_ms": ttft,
                        "http_status": r.status_code,
                        "error": ERROR_STREAM, "detail": err}
            if saw_end or got_delta:
                return {"layer": "inference", "model": model, "ok": True,
                        "latency_ms": latency, "ttft_ms": ttft,
                        "http_status": r.status_code,
                        "error": None, "detail": ""}
            if parse_lines == 0:
                return {"layer": "inference", "model": model, "ok": False,
                        "latency_ms": latency, "ttft_ms": ttft,
                        "http_status": r.status_code,
                        "error": ERROR_PARSE,
                        "detail": "流式响应不是 SSE（无 data 事件）"}
            # 已收到内容但未收尾 = 断流（stream_truncated）；只有心跳/空块 → 提前结束
            if first_token_ts or got_delta:
                return {"layer": "inference", "model": model, "ok": False,
                        "latency_ms": latency, "ttft_ms": ttft,
                        "http_status": r.status_code,
                        "error": ERROR_STREAM,
                        "detail": "stream_truncated: 已开始生成但未收到 %s"
                        % ("[DONE]" if not is_anthropic else "message_stop")}
            return {"layer": "inference", "model": model, "ok": False,
                    "latency_ms": latency, "ttft_ms": ttft,
                    "http_status": r.status_code,
                    "error": ERROR_NETWORK,
                    "detail": "流过早结束，未收到任何内容且无收尾事件"}
    except Exception as e:
        latency = int((time.monotonic() - t0) * 1000)
        return {"layer": "inference", "model": model, "ok": False,
                "latency_ms": latency, "ttft_ms": None,
                "http_status": None,
                "error": classify_error(None, e), "detail": str(e)[:400]}


async def probe_inference(client: httpx.AsyncClient, target: dict, timeout: float,
                          model: str) -> dict:
    # 流式类型（*-stream）走 SSE 探测
    if target["type"].endswith("-stream"):
        return await probe_inference_stream(client, target, timeout, model)
    t0 = time.monotonic()
    prompt = (target.get("prompt") or "").strip()
    msg_content = prompt if prompt else "ping"
    try:
        if target["type"].startswith("anthropic"):
            url = target["base_url"].rstrip("/") + "/v1/messages"
            headers = {"anthropic-version": "2023-06-01",
                       "content-type": "application/json"}
            if target.get("api_key"):
                headers["x-api-key"] = target["api_key"]
            max_tok = 4096 if prompt else 64
            body = {"model": model, "max_tokens": max_tok,
                    "messages": [{"role": "user", "content": msg_content}]}
        else:
            url = build_v1_url(target["base_url"], "/chat/completions")
            headers = {"content-type": "application/json"}
            if target.get("api_key"):
                headers["authorization"] = f"Bearer {target['api_key']}"
            body = {"model": model,
                    "messages": [{"role": "user", "content": msg_content}],
                    "stream": False}
            if not prompt:
                body["max_tokens"] = 64
        body = _merge_extra_body(body, target)
        r = await client.post(url, json=body, headers=headers, timeout=timeout)
        latency = int((time.monotonic() - t0) * 1000)
        err = classify_error(r.status_code, None)
        detail = ""
        if err is None and r.status_code < 400:
            try:
                data = r.json()
                if not (isinstance(data, dict) and ("choices" in data or "content" in data)):
                    err = ERROR_PARSE
            except Exception:
                err = ERROR_PARSE
        if r.status_code >= 400:
            try:
                detail = r.json()
                if isinstance(detail, dict):
                    detail = str(detail.get("error", {}).get("message", detail))
                else:
                    detail = str(detail)
                detail = detail[:400]
            except Exception:
                detail = r.text[:400] if r.text else ""
        return {"layer": "inference", "model": model, "ok": err is None,
                "latency_ms": latency, "ttft_ms": None,
                "http_status": r.status_code, "error": err, "detail": detail}
    except Exception as e:
        latency = int((time.monotonic() - t0) * 1000)
        return {"layer": "inference", "model": model, "ok": False,
                "latency_ms": latency, "ttft_ms": None,
                "http_status": None,
                "error": classify_error(None, e), "detail": str(e)[:400]}


class Engine:
    """探测调度：每 15s tick 一次，找出到期 target 并发探测（semaphore 限流）。"""

    def __init__(self, max_concurrent: int = 4, max_model_concurrent: int = 4):
        self.sem = asyncio.Semaphore(max_concurrent)
        self.model_sem = asyncio.Semaphore(max_model_concurrent)
        self._inflight: set[int] = set()
        self._last_tick_ts: float = 0.0

    async def tick(self) -> None:
        t0 = time.time()
        conn = core.get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM targets WHERE (enabled=1 AND show_on_status=1)"
                " OR auto_disabled=1").fetchall()
            now = time.time()
            due = []  # (target, slot)
            for row in rows:
                t = dict(row)
                if t["id"] in self._inflight:
                    continue
                # 自动禁用目标按试探间隔轻量检查（/models 是否恢复），不用原探测间隔
                if t["auto_disabled"]:
                    rr_val = t.get("suspend_retry_seconds")
                    retry_sec = int(rr_val) if rr_val is not None else int(
                        core.get_setting("suspend_retry_seconds", "1800"))
                    interval = retry_sec if retry_sec > 0 else t["interval_seconds"]
                else:
                    interval = t["interval_seconds"]
                # 绝对网格对齐：探测锚定在绝对时钟网格上（120s → 整 2 分钟）。
                # last_check_ts 存网格点而非触发时刻，实际间隔 = 配置间隔 ± tick 粒度
                # （平均正好 = 配置值），不再从"上次探测结束"起算导致间隔被拉长
                # （旧逻辑实际间隔 ~129s > 120s，色块条相位漂移 → 周期性空桶）。
                slot = int(now / interval) * interval
                if (t["last_check_ts"] or 0) < slot:
                    t["api_key"] = core.decrypt_api_key(t["api_key_enc"])
                    due.append((t, slot))
                    self._inflight.add(t["id"])
            # 占位 last_check_ts = 对齐后的网格点：探测开始前就更新，避免 tick
            # 阻塞期间 APScheduler 跳过后续 tick 导致探测周期被拉长。
            if due:
                conn.executemany(
                    "UPDATE targets SET last_check_ts=? WHERE id=?",
                    [(slot, t["id"]) for t, slot in due])
                conn.commit()
        finally:
            conn.close()
        await self._maybe_cleanup()
        # 注：故障置顶不再按 TTL 撤除——挂到目标变绿为止（清理逻辑见 notify）
        if not due:
            # 看门狗：tick 卡顿探测（上次 tick 超过 3 个周期仍无 due 视为异常）
            if self._inflight and time.time() - self._last_tick_ts > 20:
                print(f"[probe] WATCHDOG: inflight={len(self._inflight)} "
                      f"({time.time() - self._last_tick_ts:.0f}s since last due tick)")
            self._last_tick_ts = time.time()
            return
        # 后台执行探测：tick 立即返回，调度器保持 tick 粒度
        asyncio.create_task(self._run_due(due))

    async def _run_due(self, due: list[tuple[dict, float]]) -> None:
        """并发探测到期目标（tick 后台任务，独立于调度器）。"""
        t0 = time.time()
        try:
            res = await asyncio.gather(
                *(asyncio.create_task(self._check_one(t, start_at=slot))
                  for t, slot in due),
                return_exceptions=True)
            errs = [r for r in res if isinstance(r, Exception)]
            if errs:
                print(f"[probe] _run_due {len(errs)}/{len(res)} errors: "
                      f"{[str(e)[:200] for e in errs]}")
        finally:
            for t, _ in due:
                self._inflight.discard(t["id"])
            print(f"[probe] _run_due done: {len(due)} targets, "
                  f"{time.time() - t0:.1f}s, inflight={len(self._inflight)}")

    async def _maybe_cleanup(self) -> None:
        """定期清理过期 checks 与已解决 incidents，防止数据库无限膨胀。

        间隔 CLEANUP_INTERVAL（6h）检查一次；保留期默认 7 天，
        可通过 settings.checks_retention_days 调整。清理失败不影响探测。
        """
        try:
            conn = core.get_conn()
            try:
                row = conn.execute(
                    "SELECT value FROM settings WHERE key='last_cleanup_ts'").fetchone()
                last = float(row["value"]) if row else 0.0
                if time.time() - last < CLEANUP_INTERVAL:
                    return
                days = int(core.get_setting("checks_retention_days", "7"))
                c1 = conn.execute(
                    "DELETE FROM checks WHERE checked_at < datetime('now', ?)",
                    (f"-{days} days",)).rowcount
                c2 = conn.execute(
                    "DELETE FROM incidents WHERE ended_at IS NOT NULL"
                    " AND ended_at < datetime('now','-30 days')").rowcount
                conn.execute(
                    "INSERT OR REPLACE INTO settings(key,value) VALUES('last_cleanup_ts',?)",
                    (str(time.time()),))
                conn.commit()
                if c1 or c2:
                    print(f"[cleanup] removed {c1} checks, {c2} incidents "
                          f"(retention {days}d)")
            finally:
                conn.close()
        except Exception as e:
            print(f"[cleanup] failed: {e}")

    async def _resolve_models(self, client: httpx.AsyncClient, t: dict,
                              timeout: float) -> tuple[list[str], str]:
        """解析本轮要探测的模型列表。返回 (模型列表, 模式)。

        mode: explicit（显式列表）/ auto（/models 动态读取）/ auto_failed（读取失败）
        动态模式读取成功后对比快照：快照有本次无 → 标记移除（model_not_found 记录），
        本次有快照无 → 新增模型（自动进入探测列表）。/models 请求失败不更新快照。
        """
        explicit = [m.strip() for m in (t.get("model_name") or "").split(",")
                    if m.strip()]
        if explicit:
            self._mark_explicit_removed(t["id"], explicit)
            return explicit, "explicit"
        try:
            url = build_v1_url(t["base_url"], "/models")
            headers = {"authorization": f"Bearer {t['api_key']}"} if t.get("api_key") else {}
            r = await client.get(url, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                models = [m["id"] for m in data.get("data", [])
                          if isinstance(m, dict) and m.get("id")]
                if len(models) > MAX_AUTO_MODELS:
                    models = models[:MAX_AUTO_MODELS]
                self._update_snapshot(t["id"], models)
                return models, "auto"
            return [], "auto_failed"
        except Exception:
            return [], "auto_failed"

    def _update_snapshot(self, target_id: int, models: list[str]) -> None:
        """对比 /models 快照：记录被移除的模型（写 model_not_found check）。"""
        conn = core.get_conn()
        try:
            row = conn.execute(
                "SELECT model_snapshot FROM targets WHERE id=?",
                (target_id,)).fetchone()
            old = set()
            if row and row["model_snapshot"]:
                try:
                    old = set(json.loads(row["model_snapshot"]))
                except Exception:
                    old = set()
            new = set(models)
            removed = old - new
            now = core.now_iso()
            for m in removed:
                conn.execute(
                    "INSERT INTO checks(target_id,layer,model,ok,latency_ms,http_status,"
                    " error,detail,checked_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (target_id, "inference", m, 0, None, None, "model_not_found",
                     "模型从 /models 消失（上游移除）", now))
            conn.execute(
                "UPDATE targets SET model_snapshot=? WHERE id=?",
                (json.dumps(models), target_id))
            conn.commit()
        finally:
            conn.close()

    def _mark_explicit_removed(self, target_id: int, explicit: list[str]) -> None:
        """显式模式：把 checks 里有记录但不在显式列表的模型标记为移除。

        复用动态模式的 model_not_found 移除机制——展示层 main.py 的 removed
        过滤会自动隐藏这些模型。只写标记、不删数据；标记记录不进探测
        results，不触发任何通知（notify 对 model_not_found 显式排除，
        不计故障、不进事件原因）。同一模型只写一次，避免每轮重复插入。
        """
        conn = core.get_conn()
        try:
            rows = conn.execute(
                "SELECT DISTINCT model FROM checks"
                " WHERE target_id=? AND layer='inference' AND model != ''",
                (target_id,)).fetchall()
            seen = {r["model"] for r in rows} - set(explicit)
            if not seen:
                return
            now = core.now_iso()
            for m in seen:
                last = conn.execute(
                    "SELECT error FROM checks WHERE target_id=? AND layer='inference'"
                    " AND model=? ORDER BY id DESC LIMIT 1",
                    (target_id, m)).fetchone()
                if last and last["error"] == "model_not_found":
                    continue  # 已标记，避免每轮重复插入
                conn.execute(
                    "INSERT INTO checks(target_id,layer,model,ok,latency_ms,http_status,"
                    " error,detail,checked_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (target_id, "inference", m, 0, None, None, "model_not_found",
                     "模型不在显式列表，自动剔除显示", now))
            conn.commit()
        finally:
            conn.close()

    async def _run_probes(self, t: dict) -> list[dict]:
        """执行探测，返回各层结果。每条 inference 记录对应一个模型。"""
        timeout = max(1, t["timeout_seconds"])
        async with httpx.AsyncClient(timeout=timeout) as client:
            results = []
            if t["probe_mode"] in ("inference", "both"):
                models, mode = await self._resolve_models(client, t, timeout)
                if mode == "auto_failed":
                    results.append({"layer": "inference", "model": "",
                                    "ok": False, "latency_ms": 0,
                                    "http_status": None,
                                    "error": "models_fetch_failed",
                                    "detail": "无法从 /models 读取模型列表"})
                elif mode == "auto" and not models:
                    # 上游 /models 正常响应但模型列表为空 → 目标不可用：
                    # 直接短路返回失败（跳过连通性探测，避免 connectivity 成功
                    # 把整轮拉回 partial/up），连续失败达到阈值后进入 down 并告警。
                    results.append({"layer": "inference", "model": "",
                                    "ok": False, "latency_ms": 0,
                                    "http_status": 200,
                                    "error": "no_models_available",
                                    "detail": "上游 /models 返回空列表（无可用模型）"})
                    return results
                else:
                    now = time.time()
                    # 停测参数：目标级优先（0 = 每轮都探测），留空用全局默认（settings）
                    sf_val = t.get("suspend_fails")
                    suspend_fails = int(sf_val) if sf_val is not None else int(
                        core.get_setting("model_suspend_fails", "5"))
                    rr_val = t.get("suspend_retry_seconds")
                    retry_seconds = int(rr_val) if rr_val is not None else int(
                        core.get_setting("suspend_retry_seconds", "1800"))
                    conn = core.get_conn()
                    try:
                        tasks = []
                        for m in models:
                            # 停测模型跳过探测，到试探间隔才发一次；
                            # 例外：模型刚重新出现在 /models（上次是移除记录）→ 立即恢复探测
                            if is_model_suspended(conn, t["id"], m, suspend_fails):
                                last = conn.execute(
                                    "SELECT error, checked_at FROM checks"
                                    " WHERE target_id=? AND layer='inference' AND model=?"
                                    " ORDER BY id DESC LIMIT 1",
                                    (t["id"], m)).fetchone()
                                if not (last and last["error"] == "model_not_found"):
                                    last_ts = core.parse_iso(last["checked_at"]) if last else 0
                                    if retry_seconds > 0 and now - last_ts < retry_seconds:
                                        continue
                            tasks.append(self._probe_model(client, t, timeout, m))
                        if tasks:
                            results.extend(await asyncio.gather(*tasks))
                    finally:
                        conn.close()
            if t["probe_mode"] in ("connectivity", "both"):
                results.append(await probe_connectivity(client, t, timeout))
        return results
    async def _probe_model(self, client: httpx.AsyncClient, t: dict,
                           timeout: float, m: str) -> dict:
        """单个模型探测（模型级并发，model_sem 限流）。"""
        async with self.model_sem:
            return await probe_inference(client, t, timeout, m)

    async def _check_one(self, t: dict, update_state: bool = True,
                         start_at: float | None = None) -> list[dict]:
        async with self.sem:
            results = await self._run_probes(t)

        conn = core.get_conn()
        try:
            in_maintenance = self._has_maintenance(conn, t["id"])
            # start_at：调度器传网格点 slot（探测开始时刻，绝对网格对齐）。
            # checked_at 是完成时刻（探测耗时后写入）。bars 分桶用 start_at，
            # 与分桶网格同源对齐，避免探测耗时波动造成相位漂移。
            start_iso = (time.strftime("%Y-%m-%d %H:%M:%S",
                                       time.gmtime(start_at))
                         if start_at is not None else None)
            # 模型级通知计算：本轮失败即收集（是否首次告警/恢复对称性由
            # sync_target_notifications 内部判定：已告警未恢复不重复推）；
            # 上轮失败 → 本轮成功 = 恢复。同类别同一轮合并成一条推送。
            new_failed = []
            new_recovered = []
            if update_state and not in_maintenance:
                for r in results:
                    if r["layer"] != "inference" or not r.get("model"):
                        continue
                    prev = conn.execute(
                        "SELECT ok FROM checks WHERE target_id=? AND layer='inference'"
                        " AND model=? ORDER BY id DESC LIMIT 1",
                        (t["id"], r["model"])).fetchone()
                    if not r["ok"]:
                        err = r["error"] or f"HTTP {r['http_status']}"
                        new_failed.append((r["model"], err))
                    elif r["ok"] and prev is not None and prev["ok"] == 0:
                        new_recovered.append((r["model"], r["latency_ms"]))

            for r in results:
                conn.execute(
                    "INSERT INTO checks(target_id,layer,model,ok,latency_ms,ttft_ms,http_status,error,detail,checked_at,start_at)"
                    " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (t["id"], r["layer"], r.get("model", ""), 1 if r["ok"] else 0,
                     r["latency_ms"], r.get("ttft_ms"), r["http_status"], r["error"],
                     r["detail"], core.now_iso(), start_iso))

            if not update_state:
                conn.commit()
                return results

            # 轮级状态判定：全部成功 → up；全部失败 → down 候选；混合 → up（只要可用）
            oks = [r["ok"] for r in results]
            # 空 results（如全部模型停测且未到试探间隔）绝不能判为可用：
            # all([]) == True 会把"没有任何探测结果"误判成全部成功。
            all_ok = bool(oks) and all(oks)
            all_fail = not oks or not any(oks)
            # 延迟 = 该轮全部成功探测的均值
            ok_lats = [r["latency_ms"] for r in results
                       if r["ok"] and r["latency_ms"] is not None]
            latency = int(sum(ok_lats) / len(ok_lats)) if ok_lats else None
            failed = [r for r in results if not r["ok"]]
            err_text = None
            if failed:
                parts = []
                for r in failed:
                    if r.get("error") == "no_models_available":
                        parts.append("上游 /models 返回空列表（无可用模型）")
                        continue
                    label = r.get("model") or r["layer"]
                    parts.append(f"{label}:{r['error'] or ('HTTP ' + str(r['http_status']))}")
                err_text = ", ".join(parts)

            status_before = t["status"]
            streak = t["streak"]
            fail_th = t["fail_threshold"]
            recover_th = t["recover_threshold"]
            # 结构性故障：上游 /models 返回空列表（无任何模型）——跳过防抖立即 down。
            # 防抖针对网络抖动；模型列表被清空是明确的破坏性变化，应第一时间告警。
            no_models = any(r.get("error") == "no_models_available" for r in results)
            if no_models:
                streak = -fail_th
            elif all_ok:
                streak = min(streak + 1, recover_th) if streak >= 0 else 1
            elif all_fail:
                streak = max(streak - 1, -fail_th) if streak <= 0 else -1
            else:
                # 部分失败：打断连续成功/失败计数
                streak = 0

            # 状态判定：全成功/部分成功 → up；连续全失败达到阈值 → down
            # 部分成功视为可用（不因部分模型慢/失败标黄），模型级告警仍会触发
            if no_models:
                status = "down"
            elif all_fail and streak <= -fail_th:
                status = "down"
            else:
                status = "up"

            conn.execute(
                "UPDATE targets SET status=?, streak=?, last_check_at=?,"
                " last_latency_ms=?, last_error=?, updated_at=? WHERE id=?",
                (status, streak, core.now_iso(), latency, err_text,
                 core.now_iso(), t["id"]))

            # 事件只在 down（连续 fail_threshold 次全失败坐实）时开立——瞬时抖动
            # （如单模型一次超时）不产生事件记录。恢复通知仅在有关事件时发，
            # 无事件的部分失败→up 视为抖动静默处理，不推无头恢复。
            # 异常通知开关：notify_enabled=0 时该目标不推送任何自动告警/恢复通知，
            # 但状态机与事件记录照常进行（仅静默通知，不影响监测/展示）。
            notify_on = bool(int(t.get("notify_enabled") or 0))
            if status == "down" and status_before != "down":
                # up/unknown/历史 partial → down：开事件（状态机照旧，通知由下方色条判定驱动）
                existing = conn.execute(
                    "SELECT id FROM incidents WHERE target_id=? AND status='ongoing'"
                    " ORDER BY id DESC LIMIT 1", (t["id"],)).fetchone()
                if existing:
                    conn.execute(
                        "UPDATE incidents SET severity=?, note=? WHERE id=?",
                        (status, err_text, existing["id"]))
                else:
                    conn.execute(
                        "INSERT INTO incidents(target_id,started_at,status,severity,note,created_at)"
                        " VALUES(?,?,?,?,?,?)",
                        (t["id"], core.now_iso(), "ongoing", status, err_text, core.now_iso()))
            elif status == "up" and status_before in ("partial", "down"):
                row = conn.execute(
                    "SELECT id, started_at FROM incidents WHERE target_id=? AND status='ongoing'"
                    " ORDER BY id DESC LIMIT 1", (t["id"],)).fetchone()
                if row:
                    dur = int(time.time() - core.parse_iso(row["started_at"]))
                    conn.execute(
                        "UPDATE incidents SET ended_at=?, duration_seconds=?, status='resolved'"
                        " WHERE id=?",
                        (core.now_iso(), dur, row["id"]))
            # 通知同步（与状态机事件解耦，按本轮色条结果即时判定）：
            # 全红 → 分组故障置顶（不列模型明细）；部分红 → 模型级增量（新红模型并入置顶、
            # 恢复模型发 30s 通知并移出置顶）；全绿 → 变绿通知（30s 删）+ 撤下故障置顶。
            fault_all = bool(oks) and not any(oks)
            if not in_maintenance and notify_on:
                asyncio.create_task(
                    notify.sync_target_notifications(
                        t["name"], t["id"], new_failed, new_recovered,
                        fault_all, all_ok, err_text))

            # 自动禁用/恢复：上游无模型 → 自动禁用并隐藏（区分手动/自动禁用，
            # auto_disabled=1 的目标由 tick 按试探间隔轻量检查，恢复后自动重新启用）。
            if no_models and not in_maintenance and not t.get("auto_disabled"):
                conn.execute(
                    "UPDATE targets SET enabled=0, show_on_status=0, auto_disabled=1,"
                    " last_check_ts=? WHERE id=?",
                    (time.time(), t["id"]))
            elif status == "up" and t.get("auto_disabled"):
                conn.execute(
                    "UPDATE targets SET enabled=1, show_on_status=1, auto_disabled=0,"
                    " updated_at=? WHERE id=?",
                    (core.now_iso(), t["id"]))

            conn.commit()
        finally:
            conn.close()
        return results

    @staticmethod
    def _has_maintenance(conn, target_id: int) -> bool:
        row = conn.execute(
            "SELECT 1 FROM incidents WHERE target_id=? AND status='maintenance'"
            " AND ended_at IS NULL LIMIT 1", (target_id,)).fetchone()
        return row is not None

    async def test_target(self, target_id: int) -> list[dict]:
        """管理页手动测试：真实探测并写入 checks，但不改动状态机。"""
        conn = core.get_conn()
        row = conn.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
        conn.close()
        if row is None:
            raise ValueError("target not found")
        t = dict(row)
        t["api_key"] = core.decrypt_api_key(t["api_key_enc"])
        return await self._check_one(t, update_state=False)

    async def trigger(self, target_id: int) -> None:
        """保存目标后立即探测一轮（update_state=True，正常推进状态机）。

        先占位 last_check_ts，避免调度器 tick 在同一轮重复探测。
        """
        conn = core.get_conn()
        try:
            row = conn.execute("SELECT * FROM targets WHERE id=?", (target_id,)).fetchone()
            if row is None:
                return
            conn.execute("UPDATE targets SET last_check_ts=? WHERE id=?",
                         (time.time(), target_id))
            conn.commit()
            t = dict(row)
        finally:
            conn.close()
        t["api_key"] = core.decrypt_api_key(t["api_key_enc"])
        await self._check_one(t, update_state=True)


engine = Engine()
