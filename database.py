import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import threading


# ========================= Вспомогательные =========================

def _to_iso(ts) -> Optional[str]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.isoformat(timespec="seconds")
    try:
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.utcfromtimestamp(float(ts))
            return dt.replace(microsecond=0).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_json(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return _to_iso(obj)
    if isinstance(obj, dict):
        return {k: _normalize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_json(v) for v in obj]
    return obj


# =============================== DB ===============================

class Database:
    def __init__(self, db_path: str = "kwin_bot.db", memory: bool = False):
        self.db_path = ":memory:" if memory else db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # -------------------------- Схема --------------------------
    def _init_schema(self) -> None:
        c = self.conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT,
                direction    TEXT NOT NULL,
                entry_price  REAL NOT NULL,
                exit_price   REAL,
                stop_loss    REAL,
                take_profit  REAL,
                quantity     REAL,
                qty          REAL,
                pnl          REAL,
                rr           REAL,
                entry_time   TEXT NOT NULL,
                exit_time    TEXT,
                exit_reason  TEXT,
                status       TEXT NOT NULL DEFAULT 'open',
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                equity    REAL NOT NULL,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                id         INTEGER PRIMARY KEY,
                state_data TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id          INTEGER PRIMARY KEY,
                config_data TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                level     TEXT NOT NULL,
                message   TEXT NOT NULL,
                module    TEXT,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)
        self.conn.commit()

    # -------------------------- Trades --------------------------
    def save_trade(self, trade: Dict) -> int:
        with self._lock:
            try:
                c = self.conn.cursor()
                c.execute("""
                    INSERT INTO trades (
                        symbol, direction, entry_price, exit_price, stop_loss, take_profit,
                        quantity, qty, pnl, rr, entry_time, exit_time, exit_reason, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade.get("symbol"),
                    trade.get("direction"),
                    _f(trade.get("entry_price"), 0.0),
                    _f(trade.get("exit_price")),
                    _f(trade.get("stop_loss")),
                    _f(trade.get("take_profit")),
                    _f(trade.get("quantity")),
                    _f(trade.get("qty")),
                    _f(trade.get("pnl")),
                    _f(trade.get("rr")),
                    _to_iso(trade.get("entry_time") or datetime.utcnow()),
                    _to_iso(trade.get("exit_time")),
                    trade.get("exit_reason"),
                    trade.get("status", "open"),
                ))
                self.conn.commit()
                return int(c.lastrowid or 0)
            except Exception:
                self.conn.rollback()
                raise

    def add_trade(self, trade: Dict) -> int:
        return self.save_trade(trade)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        with self._lock:
            try:
                c = self.conn.cursor()
                trade_id = trade_data.get("trade_id")
                if trade_id:
                    c.execute("SELECT * FROM trades WHERE id = ? AND status = 'open'", (trade_id,))
                else:
                    c.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC LIMIT 1")
                row = c.fetchone()
                if not row:
                    return
                tr = dict(row)
                entry_price = _f(tr.get("entry_price"), 0.0)
                stop_loss   = _f(tr.get("stop_loss"))
                qty         = _f(tr.get("quantity")) or _f(tr.get("qty"), 0.0)
                side        = tr.get("direction")
                exit_price  = _f(trade_data.get("exit_price"), entry_price)

                gross = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
                fee_in  = entry_price * qty * fee_rate
                fee_out = exit_price * qty * fee_rate
                net_pnl = gross - (fee_in + fee_out)

                rr = None
                if stop_loss is not None and stop_loss != 0:
                    risk = abs(entry_price - stop_loss)
                    if risk > 0:
                        rr = abs(exit_price - entry_price) / risk

                c.execute("""
                    UPDATE trades
                       SET exit_price  = ?,
                           exit_time   = ?,
                           exit_reason = ?,
                           pnl         = ?,
                           rr          = ?,
                           status      = ?
                     WHERE id = ?
                """, (
                    exit_price,
                    _to_iso(trade_data.get("exit_time") or datetime.utcnow()),
                    trade_data.get("exit_reason"),
                    _f(net_pnl),
                    _f(rr),
                    trade_data.get("status", "closed"),
                    tr["id"],
                ))
                self.conn.commit()
            except Exception:
                self.conn.rollback()
                raise

    def get_open_trade(self) -> Optional[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC LIMIT 1")
            row = c.fetchone()
            return dict(row) if row else None

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (int(limit),))
            return [dict(r) for r in c.fetchall()]

    def get_all_trades(self) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY entry_time ASC")
            return [dict(r) for r in c.fetchall()]

    # -------------------------- Equity --------------------------
    def save_equity_snapshot(self, equity: float):
        def _worker(eq):
            try:
                with self._lock:
                    c = self.conn.cursor()
                    c.execute("INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                              (_f(eq, 0.0), _to_iso(datetime.utcnow())))
                    self.conn.commit()
            except Exception:
                pass
        threading.Thread(target=_worker, args=(equity,), daemon=True).start()

    def get_equity_history(self, days: int = 30) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute(
                "SELECT equity, timestamp FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                (_to_iso(start),),
            )
            return [{"equity": r[0], "timestamp": r[1]} for r in c.fetchall()]

    # -------------------------- Logs --------------------------
    def save_log(self, level: str, message: str, module: str = None):
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO logs (level, message, module, timestamp) VALUES (?, ?, ?, ?)",
                (level, message, module, _to_iso(datetime.utcnow())),
            )
            self.conn.commit()

    def get_logs(self, limit: int = 500) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM logs ORDER BY id DESC LIMIT ?", (int(limit),))
            return [dict(r) for r in c.fetchall()]

    def purge_old_logs(self, keep: int = 1000):
        with self._lock:
            c = self.conn.cursor()
            c.execute("DELETE FROM logs WHERE id NOT IN (SELECT id FROM logs ORDER BY id DESC LIMIT ?)", (keep,))
            self.conn.commit()

    # --------------------- Bot State / Config ---------------------
    def get_bot_state(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT state_data FROM bot_state WHERE id = 1")
            row = c.fetchone()
            return json.loads(row[0]) if row else {}

    def save_bot_state(self, state: Dict[str, Any]):
        with self._lock:
            c = self.conn.cursor()
            c.execute("INSERT OR REPLACE INTO bot_state (id, state_data, updated_at) VALUES (1, ?, ?)",
                      (json.dumps(_normalize_json(state)), _to_iso(datetime.utcnow())))
            self.conn.commit()

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT config_data FROM config WHERE id = 1")
            row = c.fetchone()
            return json.loads(row[0]) if row else {}

    def save_config(self, cfg: Dict[str, Any]):
        with self._lock:
            c = self.conn.cursor()
            c.execute("INSERT OR REPLACE INTO config (id, config_data, updated_at) VALUES (1, ?, ?)",
                      (json.dumps(_normalize_json(cfg)), _to_iso(datetime.utcnow())))
            self.conn.commit()

    # --------------------- Экспорт/Импорт ---------------------
    def export_json(self) -> Dict[str, Any]:
        return {
            "trades": self.get_all_trades(),
            "equity": self.get_equity_history(9999),
            "state": self.get_bot_state(),
            "config": self.get_config(),
            "logs": self.get_logs(500)
        }

    def import_json(self, dump: Dict[str, Any]):
        self.drop_and_recreate()
        for tr in dump.get("trades", []):
            self.save_trade(tr)
        for e in dump.get("equity", []):
            self.save_equity_snapshot(e["equity"])
        self.save_bot_state(dump.get("state", {}))
        self.save_config(dump.get("config", {}))
        for lg in dump.get("logs", []):
            self.save_log(lg["level"], lg["message"], lg.get("module"))

    def drop_and_recreate(self):
        with self._lock:
            c = self.conn.cursor()
            c.executescript("""
                DROP TABLE IF EXISTS trades;
                DROP TABLE IF EXISTS equity_history;
                DROP TABLE IF EXISTS bot_state;
                DROP TABLE IF EXISTS config;
                DROP TABLE IF EXISTS logs;
            """)
            self.conn.commit()
        self._init_schema()

    # --------------------- Reset для Backtest ---------------------
    def reset_for_backtest(self):
        """Полностью очищает БД перед новым бэктестом"""
        with self._lock:
            c = self.conn.cursor()
            c.executescript("""
                DELETE FROM trades;
                DELETE FROM equity_history;
                DELETE FROM bot_state;
                DELETE FROM config;
                DELETE FROM logs;
            """)
            self.conn.commit()
