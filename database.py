# database.py
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import threading  # ← для потокобезопасности


# ========================= Вспомогательные =========================

def _to_iso(ts) -> Optional[str]:
    """Преобразует datetime (naive/aware/int ms) в ISO-8601 строку; пропускает None."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.isoformat(timespec="seconds")
    try:
        # поддержка timestamp (sec/ms)
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.utcfromtimestamp(float(ts))
            return dt.replace(microsecond=0).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


def _f(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _normalize_json(obj: Any) -> Any:
    """
    Рекурсивно нормализует объект для json.dumps:
    - datetime -> ISO-строка
    - dict/list -> обход внутрь
    - остальное без изменений
    """
    if isinstance(obj, datetime):
        return _to_iso(obj)
    if isinstance(obj, dict):
        return {k: _normalize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_json(v) for v in obj]
    return obj


# =============================== DB ===============================

class Database:
    """
    SQLite хранилище для бота и бэктеста.
    Совместимо с вызовами:
      - add_trade/save_trade/update_trade_exit
      - get_recent_trades/get_all_trades/get_trades_by_period
      - save_equity_snapshot/get_equity_history
      - get_performance_stats/get_trades_count_today/get_pnl_today
      - save_bot_state/get_bot_state
      - save_config/get_config
    """

    def __init__(self, db_path: str = "kwin_bot.db", memory: bool = False):
        self.db_path = ":memory:" if memory else db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()

    # -------------------------- Схема --------------------------

    def _init_schema(self) -> None:
        c = self.conn.cursor()

        # Таблица сделок
        c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol       TEXT,
                direction    TEXT NOT NULL,             -- long/short
                entry_price  REAL NOT NULL,
                exit_price   REAL,
                stop_loss    REAL,
                take_profit  REAL,
                quantity     REAL,                      -- qty в базовой
                qty          REAL,                      -- алиас для совместимости
                pnl          REAL,
                rr           REAL,
                entry_time   TEXT NOT NULL,
                exit_time    TEXT,
                exit_reason  TEXT,
                status       TEXT NOT NULL DEFAULT 'open',
                created_at   TEXT DEFAULT (datetime('now'))
            )
        """)

        # История equity
        c.execute("""
            CREATE TABLE IF NOT EXISTS equity_history (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                equity    REAL NOT NULL,
                timestamp TEXT DEFAULT (datetime('now'))
            )
        """)

        # Состояние бота (единственная строка id=1)
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                id         INTEGER PRIMARY KEY,
                state_data TEXT NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)

        # Конфигурация (единственная строка id=1)
        c.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id          INTEGER PRIMARY KEY,
                config_data TEXT NOT NULL,
                updated_at  TEXT DEFAULT (datetime('now'))
            )
        """)

        # Логи
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
        """
        Вставка сделки. Возвращает id.
        Поддерживает как поля quantity, так и qty.
        """
        with self._lock:
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

    # alias для обратной совместимости
    def add_trade(self, trade: Dict) -> int:
        return self.save_trade(trade)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """
        Закрыть последнюю открытую сделку (или по id), посчитать PnL/RR и обновить запись.
        """
        with self._lock:
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
            qty         = _f(tr.get("quantity"), None) or _f(tr.get("qty"), 0.0)
            side        = tr.get("direction")
            exit_price  = _f(trade_data.get("exit_price"), entry_price)

            # валовой pnl
            gross = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
            # комиссии (вход+выход)
            fee_in  = entry_price * qty * fee_rate
            fee_out = exit_price * qty * fee_rate
            net_pnl = gross - (fee_in + fee_out)

            rr = None
            if stop_loss is not None:
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

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?", (int(limit),))
            return [dict(r) for r in c.fetchall()]

    def get_all_trades(self) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY id ASC")
            return [dict(r) for r in c.fetchall()]

    def get_trades_by_period(self, days: int) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute("SELECT * FROM trades WHERE entry_time >= ? ORDER BY entry_time DESC", (_to_iso(start),))
            return [dict(r) for r in c.fetchall()]

    # -------------------------- Equity --------------------------

    def save_equity_snapshot(self, equity: float):
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                (_f(equity, 0.0), _to_iso(datetime.utcnow())),
            )
            self.conn.commit()

    def get_equity_history(self, days: int = 30) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute(
                "SELECT equity, timestamp FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                (_to_iso(start),),
            )
            return [{"equity": r[0], "timestamp": r[1]} for r in c.fetchall()]

    # -------------------------- Stats --------------------------

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute("""
                SELECT 
                    COUNT(*)                                          AS total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END)               AS winning_trades,
                    AVG(pnl)                                          AS avg_pnl,
                    SUM(pnl)                                          AS total_pnl,
                    AVG(rr)                                           AS avg_rr,
                    MAX(pnl)                                          AS max_win,
                    MIN(pnl)                                          AS max_loss,
                    AVG(
                        CASE WHEN exit_time IS NOT NULL AND entry_time IS NOT NULL 
                             THEN (julianday(exit_time) - julianday(entry_time)) * 24 
                        END
                    )                                                 AS avg_hold_hours
                FROM trades
                WHERE entry_time >= ? AND status = 'closed'
            """, (_to_iso(start),))
            row = c.fetchone() or (0,) * 8
            total = row[0] or 0
            wins = row[1] or 0
            return {
                "total_trades": int(total),
                "winning_trades": int(wins),
                "losing_trades": int(total - wins),
                "win_rate": (wins / total * 100.0) if total else 0.0,
                "avg_pnl": float(row[2] or 0.0),
                "total_pnl": float(row[3] or 0.0),
                "avg_rr": float(row[4] or 0.0),
                "max_win": float(row[5] or 0.0),
                "max_loss": float(row[6] or 0.0),
                "avg_hold_time": float(row[7] or 0.0),
            }

    def get_trades_count_today(self) -> int:
        with self._lock:
            c = self.conn.cursor()
            today = datetime.utcnow().date().isoformat()
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ?", (today,))
            return int((c.fetchone() or (0,))[0])

    def get_pnl_today(self) -> float:
        with self._lock:
            c = self.conn.cursor()
            today = datetime.utcnow().date().isoformat()
            c.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE DATE(entry_time) = ? AND status = 'closed'",
                (today,),
            )
            return float((c.fetchone() or (0.0,))[0])

    # -------------------------- Logs --------------------------

    def save_log(self, level: str, message: str, module: str = None):
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO logs (level, message, module, timestamp) VALUES (?, ?, ?, ?)",
                (level, message, module, _to_iso(datetime.utcnow())),
            )
            self.conn.commit()

    def get_logs(self, limit: int = 100) -> List[Dict]:
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "SELECT level, message, module, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            )
            cols = ["level", "message", "module", "timestamp"]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    # --------------------- State & Config (НОВОЕ) ---------------------

    def save_bot_state(self, state: Dict) -> None:
        """
        Сохранение состояния бота (dict в единственной строке с id=1).
        Автоматически конвертирует datetime -> ISO, чтобы json.dumps не падал.
        """
        safe_state = _normalize_json(state or {})
        payload = json.dumps(safe_state, ensure_ascii=False)
        with self._lock:
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO bot_state (id, state_data, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    state_data = excluded.state_data,
                    updated_at = excluded.updated_at
            """, (payload, _to_iso(datetime.utcnow())))
            self.conn.commit()

    def get_bot_state(self) -> Dict:
        """
        Чтение состояния бота. Возвращает dict (или {}).
        """
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT state_data FROM bot_state WHERE id = 1")
            row = c.fetchone()
            if not row or not row[0]:
                return {}
            try:
                data = json.loads(row[0])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    def save_config(self, cfg_data: Dict) -> None:
        """
        Сохранение последней рабочей конфигурации.
        Также нормализует datetime -> ISO.
        """
        safe_cfg = _normalize_json(cfg_data or {})
        payload = json.dumps(safe_cfg, ensure_ascii=False)
        with self._lock:
            c = self.conn.cursor()
            c.execute("""
                INSERT INTO config (id, config_data, updated_at)
                VALUES (1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    config_data = excluded.config_data,
                    updated_at  = excluded.updated_at
            """, (payload, _to_iso(datetime.utcnow())))
            self.conn.commit()

    def get_config(self) -> Dict:
        """
        Получение сохранённой конфигурации (dict или {}).
        """
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT config_data FROM config WHERE id = 1")
            row = c.fetchone()
            if not row or not row[0]:
                return {}
            try:
                data = json.loads(row[0])
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

    # -------------------------- Service --------------------------

    def clear_trades(self):
        with self._lock:
            c = self.conn.cursor()
            c.execute("DELETE FROM trades")
            self.conn.commit()

    def drop_and_recreate(self):
        """Полное пересоздание схемы (на случай поврежденной БД в Railway)."""
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
