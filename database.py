# database.py
import os
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone

# ===================== Вспомогательные =====================

def _to_iso(ts) -> Optional[str]:
    """Преобразует datetime (naive/aware) в ISO-8601 строку; пропускает None."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.isoformat(timespec="seconds")
    try:
        if isinstance(ts, (int, float)):
            dt = datetime.utcfromtimestamp(float(ts) / (1000 if float(ts) > 1e12 else 1))
            return dt.replace(microsecond=0).isoformat()
        return str(ts)
    except Exception:
        return str(ts)

# Совместимость для старого кода
to_iso = _to_iso


class Database:
    """Управление базой данных SQLite для торгового бота."""

    def __init__(self, db_path: str = "kwin_bot.db", memory: bool = False):
        self.db_path = ":memory:" if memory else db_path
        self.init_database()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    # ===================== Схема =====================

    def init_database(self):
        """Инициализация базы данных (создание таблиц при отсутствии)."""
        with self._connect() as conn:
            c = conn.cursor()

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    stop_loss REAL,
                    take_profit REAL,
                    quantity REAL NOT NULL,
                    pnl REAL,
                    rr REAL,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT DEFAULT (datetime('now'))
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    timestamp TEXT DEFAULT (datetime('now'))
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY,
                    config_data TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
                """
            )

            c.execute(
                """
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    timestamp TEXT DEFAULT (datetime('now'))
                )
                """
            )

            conn.commit()

    # ===================== Trades =====================

    def save_trade(self, trade_data: Dict) -> int:
        """Сохранение новой сделки. Возвращает id."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT INTO trades (
                    symbol, direction, entry_price, stop_loss, take_profit,
                    quantity, entry_time, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_data.get("symbol"),
                    trade_data.get("direction"),
                    float(trade_data.get("entry_price")),
                    None if trade_data.get("stop_loss") is None else float(trade_data.get("stop_loss")),
                    None if trade_data.get("take_profit") is None else float(trade_data.get("take_profit")),
                    float(trade_data.get("quantity")),
                    _to_iso(trade_data.get("entry_time") or datetime.utcnow()),
                    trade_data.get("status", "open"),
                ),
            )
            trade_id = c.lastrowid or 0
            conn.commit()
            return int(trade_id)

    # Backward-compat: старое имя
    def add_trade(self, trade_data: Dict) -> int:
        return self.save_trade(trade_data)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """Закрыть открытую сделку: рассчитать PnL/RR и обновить запись."""
        with self._connect() as conn:
            c = conn.cursor()

            trade_id = trade_data.get("trade_id")
            if trade_id:
                c.execute("SELECT * FROM trades WHERE id = ? AND status = 'open'", (trade_id,))
            else:
                c.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC LIMIT 1")

            row = c.fetchone()
            if not row:
                print("No open trade found to update")
                return

            cols = [d[0] for d in c.description]
            tr = dict(zip(cols, row))

            entry_price = float(tr["entry_price"])
            stop_loss = float(tr["stop_loss"]) if tr["stop_loss"] is not None else None
            qty = float(tr["quantity"])
            side = tr["direction"]
            exit_price = float(trade_data.get("exit_price", 0.0))

            gross = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
            fee_in = entry_price * qty * fee_rate
            fee_out = exit_price * qty * fee_rate
            net_pnl = gross - (fee_in + fee_out)

            rr = 0.0
            if stop_loss is not None:
                risk_per_unit = abs(entry_price - stop_loss)
                if risk_per_unit > 0:
                    rr = abs(exit_price - entry_price) / risk_per_unit

            c.execute(
                """
                UPDATE trades
                SET exit_price = ?,
                    exit_time = ?,
                    exit_reason = ?,
                    pnl = ?,
                    rr = ?,
                    status = ?
                WHERE id = ?
                """,
                (
                    exit_price,
                    _to_iso(trade_data.get("exit_time") or datetime.utcnow()),
                    trade_data.get("exit_reason"),
                    float(net_pnl),
                    float(rr),
                    trade_data.get("status", "closed"),
                    tr["id"],
                ),
            )
            conn.commit()

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Последние N сделок (по entry_time)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (int(limit),),
            )
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    def get_all_trades(self) -> List[Dict]:
        """Все сделки (удобно для отчётов/бэктеста)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY entry_time ASC")
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    def get_trades_by_period(self, days: int) -> List[Dict]:
        """Сделки за последние days дней."""
        with self._connect() as conn:
            c = conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute(
                "SELECT * FROM trades WHERE entry_time >= ? ORDER BY entry_time DESC",
                (_to_iso(start),),
            )
            cols = [d[0] for d in c.description]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    # ===================== Equity =====================

    def save_equity_snapshot(self, equity: float):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                (float(equity), _to_iso(datetime.utcnow())),
            )
            conn.commit()

    def get_equity_history(self, days: int = 30) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute(
                "SELECT equity, timestamp FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                (_to_iso(start),),
            )
            rows = c.fetchall()
            return [{"equity": row[0], "timestamp": row[1]} for row in rows]

    # ===================== Bot state / Config =====================

    def save_bot_state(self, state: Dict):
        with self._connect() as conn:
            c = conn.cursor()
            state_json = json.dumps(state)
            c.execute(
                """
                INSERT OR REPLACE INTO bot_state (id, state_data, updated_at)
                VALUES (1, ?, ?)
                """,
                (state_json, _to_iso(datetime.utcnow())),
            )
            conn.commit()

    def get_bot_state(self) -> Optional[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT state_data FROM bot_state WHERE id = 1")
            row = c.fetchone()
            return json.loads(row[0]) if row else None

    # ===================== Stats / Reports =====================

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        with self._connect() as conn:
            c = conn.cursor()
            start = datetime.utcnow() - timedelta(days=int(days))
            c.execute(
                """
                SELECT 
                    COUNT(*) as total_trades,
                    COUNT(CASE WHEN pnl > 0 THEN 1 END) as winning_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    AVG(rr) as avg_rr,
                    MAX(pnl) as max_win,
                    MIN(pnl) as max_loss,
                    AVG(
                        CASE 
                          WHEN exit_time IS NOT NULL AND entry_time IS NOT NULL 
                          THEN (julianday(exit_time) - julianday(entry_time)) * 24 
                        END
                    ) as avg_hold_hours
                FROM trades 
                WHERE entry_time >= ? AND status = 'closed'
                """,
                (_to_iso(start),),
            )
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
        with self._connect() as conn:
            c = conn.cursor()
            today = datetime.utcnow().date().isoformat()
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = ?", (today,))
            return int((c.fetchone() or (0,))[0])

    def get_pnl_today(self) -> float:
        with self._connect() as conn:
            c = conn.cursor()
            today = datetime.utcnow().date().isoformat()
            c.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE DATE(entry_time) = ? AND status = 'closed'",
                (today,),
            )
            return float((c.fetchone() or (0.0,))[0])

    # ===================== Logs =====================

    def save_log(self, level: str, message: str, module: str = None):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (level, message, module, timestamp) VALUES (?, ?, ?, ?)",
                (level, message, module, _to_iso(datetime.utcnow())),
            )
            conn.commit()

    def get_logs(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT level, message, module, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?",
                (int(limit),),
            )
            cols = ["level", "message", "module", "timestamp"]
            rows = c.fetchall()
            return [dict(zip(cols, r)) for r in rows]

    # ===================== Cleanup =====================

    def cleanup_old_data(self, days_to_keep: int = 90):
        with self._connect() as conn:
            c = conn.cursor()
            cutoff_iso = _to_iso(datetime.utcnow() - timedelta(days=int(days_to_keep)))
            c.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_iso,))
            c.execute("DELETE FROM equity_history WHERE timestamp < ?", (cutoff_iso,))
            conn.commit()
