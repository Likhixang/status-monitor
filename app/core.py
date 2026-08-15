"""核心层：数据库、密钥加密、管理员认证。"""
import base64
import calendar
import hashlib
import hmac
import json
import os
import sqlite3
import time
from pathlib import Path

from cryptography.fernet import Fernet

DATA_DIR = Path(os.environ.get("STATUS_DATA_DIR", "/app/data"))
DB_PATH = DATA_DIR / "status.db"
KEY_FILE = DATA_DIR / "secret.key"

SCHEMA = """
CREATE TABLE IF NOT EXISTS targets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    group_name TEXT NOT NULL DEFAULT '默认',
    type TEXT NOT NULL DEFAULT 'openai',
    base_url TEXT NOT NULL,
    api_key_enc TEXT,
    model_name TEXT,
    probe_mode TEXT NOT NULL DEFAULT 'both',
    interval_seconds INTEGER NOT NULL DEFAULT 300,
    timeout_seconds INTEGER NOT NULL DEFAULT 15,
    fail_threshold INTEGER NOT NULL DEFAULT 3,
    recover_threshold INTEGER NOT NULL DEFAULT 3,
    enabled INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL DEFAULT 'unknown',
    streak INTEGER NOT NULL DEFAULT 0,
    last_check_ts REAL NOT NULL DEFAULT 0,
    last_check_at TEXT,
    last_latency_ms INTEGER,
    last_error TEXT,
    model_snapshot TEXT DEFAULT '',
    suspend_fails INTEGER,
    suspend_retry_seconds INTEGER,
    show_on_status INTEGER NOT NULL DEFAULT 1,
    auto_disabled INTEGER NOT NULL DEFAULT 0,
    sort_order INTEGER NOT NULL DEFAULT 0,
    extra_body TEXT DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    layer TEXT NOT NULL,
    model TEXT DEFAULT '',
    ok INTEGER NOT NULL,
    latency_ms INTEGER,
    ttft_ms INTEGER,
    http_status INTEGER,
    error TEXT,
    detail TEXT,
    checked_at TEXT NOT NULL,
    start_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_checks_target_time ON checks(target_id, checked_at);

CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL REFERENCES targets(id) ON DELETE CASCADE,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    duration_seconds INTEGER,
    status TEXT NOT NULL DEFAULT 'ongoing',
    severity TEXT NOT NULL DEFAULT 'partial',
    note TEXT DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_incidents_target ON incidents(target_id);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS pin_state (
    target_id INTEGER PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    message_id INTEGER NOT NULL,
    pinned_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notification_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id INTEGER NOT NULL,
    kind TEXT NOT NULL,
    model TEXT DEFAULT '',
    sent_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_notification_log_target
    ON notification_log(target_id, kind, sent_at);
"""


def now_iso() -> str:
    """UTC 时间，格式与 SQLite datetime('now') 一致（YYYY-MM-DD HH:MM:SS）。"""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())


def parse_iso(iso: str) -> float:
    """按 UTC 解析，返回 epoch 秒。"""
    return calendar.timegm(time.strptime(iso, "%Y-%m-%d %H:%M:%S"))


def get_conn() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = get_conn()
    conn.executescript(SCHEMA)
    # 旧库迁移：checks.model 列
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(checks)").fetchall()}
    if "model" not in cols:
        conn.execute("ALTER TABLE checks ADD COLUMN model TEXT DEFAULT ''")
    tcols = {r["name"] for r in conn.execute("PRAGMA table_info(targets)").fetchall()}
    if "model_snapshot" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN model_snapshot TEXT DEFAULT ''")
    if "auto_disabled" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN auto_disabled INTEGER NOT NULL DEFAULT 0")
    if "suspend_fails" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN suspend_fails INTEGER")
    if "suspend_retry_seconds" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN suspend_retry_seconds INTEGER")
    if "show_on_status" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN show_on_status INTEGER NOT NULL DEFAULT 1")
    if "sort_order" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN sort_order INTEGER NOT NULL DEFAULT 0")
    if "extra_body" not in tcols:
        conn.execute("ALTER TABLE targets ADD COLUMN extra_body TEXT DEFAULT ''")
    # 旧库迁移：incidents.severity 列（事件等级：partial 部分故障 / down 异常）
    icols = {r["name"] for r in conn.execute("PRAGMA table_info(incidents)").fetchall()}
    if "severity" not in icols:
        conn.execute("ALTER TABLE incidents ADD COLUMN severity TEXT NOT NULL DEFAULT 'partial'")
    # 旧库迁移：checks.start_at 列（探测开始时刻，网格对齐分桶用；旧记录为 NULL 回退 checked_at）
    ccols = {r["name"] for r in conn.execute("PRAGMA table_info(checks)").fetchall()}
    if "start_at" not in ccols:
        conn.execute("ALTER TABLE checks ADD COLUMN start_at TEXT")
    if "ttft_ms" not in ccols:
        conn.execute("ALTER TABLE checks ADD COLUMN ttft_ms INTEGER")
    # 旧库迁移：notification_log.model 列（模型级告警对称性检查用）
    ncols = {r["name"] for r in conn.execute("PRAGMA table_info(notification_log)").fetchall()}
    if "model" not in ncols:
        conn.execute("ALTER TABLE notification_log ADD COLUMN model TEXT DEFAULT ''")
    conn.commit()
    conn.close()

    admin_user = os.environ.get("ADMIN_USERNAME")
    admin_pass = os.environ.get("ADMIN_PASSWORD")
    if admin_user and admin_pass:
        conn = get_conn()
        row = conn.execute(
            "SELECT value FROM settings WHERE key='admin_username'").fetchone()
        if row is None:
            conn.execute(
                "INSERT OR REPLACE INTO settings(key,value) VALUES('admin_username',?)",
                (admin_user,))
            conn.execute(
                "INSERT OR REPLACE INTO settings(key,value) VALUES('admin_password_hash',?)",
                (hash_password(admin_pass),))
            conn.commit()
        conn.close()


def get_setting(key: str, default: str = "") -> str:
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_setting(key: str, value: str) -> None:
    conn = get_conn()
    conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES(?,?)", (key, value))
    conn.commit()
    conn.close()


# ---------- API key 加密 ----------

def get_cipher() -> Fernet:
    KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(Fernet.generate_key())
        KEY_FILE.chmod(0o600)
    return Fernet(KEY_FILE.read_bytes())


def encrypt_api_key(plain: str) -> str | None:
    if not plain:
        return None
    return get_cipher().encrypt(plain.encode()).decode()


def decrypt_api_key(enc: str | None) -> str | None:
    if not enc:
        return None
    try:
        return get_cipher().decrypt(enc.encode()).decode()
    except Exception:
        return None


def mask_key(key: str | None) -> str:
    """API Key 掩码：前6位 + ... + 尾4位（如 sk-abc...x9f2），用于编辑表单回显辨识。"""
    if not key:
        return ""
    if len(key) <= 12:
        return key[:4] + "***"
    return key[:6] + "..." + key[-4:]


# ---------- 管理员认证 ----------

def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000)
    return f"pbkdf2${base64.b64encode(salt).decode()}${base64.b64encode(dk).decode()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _algo, salt_b64, dk_b64 = stored.split("$")
        salt = base64.b64decode(salt_b64)
        dk = base64.b64decode(dk_b64)
        return hmac.compare_digest(
            hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000), dk)
    except Exception:
        return False


def _token_secret() -> str:
    conn = get_conn()
    row = conn.execute("SELECT value FROM settings WHERE key='token_secret'").fetchone()
    if row is None:
        sec = base64.urlsafe_b64encode(os.urandom(32)).decode()
        conn.execute("INSERT OR REPLACE INTO settings(key,value) VALUES('token_secret',?)",
                     (sec,))
        conn.commit()
        conn.close()
        return sec
    conn.close()
    return row["value"]


def issue_token(username: str, ttl_hours: int = 12) -> str:
    payload = base64.urlsafe_b64encode(
        json.dumps({"u": username, "e": time.time() + ttl_hours * 3600}).encode()
    ).decode().rstrip("=")
    sig = base64.urlsafe_b64encode(
        hmac.new(_token_secret().encode(), payload.encode(), hashlib.sha256).digest()
    ).decode().rstrip("=")
    return f"{payload}.{sig}"


def verify_token(token: str) -> str | None:
    try:
        payload_b64, sig_b64 = token.split(".")
        sig = base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4))
        expect = hmac.new(_token_secret().encode(), payload_b64.encode(),
                          hashlib.sha256).digest()
        if not hmac.compare_digest(sig, expect):
            return None
        payload = json.loads(
            base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
        if payload["e"] < time.time():
            return None
        return payload["u"]
    except Exception:
        return None
