# database.py
import os
import sqlite3
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone

# ===================== Вспомогательные =====================

def _to_iso(ts) -> Optional[str]:
    """Преобразует datetime (naive/aware) в ISO-8601 строку; пропускает None."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        # приводим к UTC и убираем tzinfo для единообразия хранения
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.isoformat(timespec="seconds")
    return str(ts)

# ===========================================================


class Database:
    """Управление базой данных SQLite для торгового бота"""

    def __init__(self, db_path: Optional[str] = None, memory: bool = False):
        # Размещение БД: :memory: | ENV DB_PATH | локальный файл
        if memory:
            self.db_path = ":memory:"
        else:
            self.db_path = db_path or os.getenv("DB_PATH", "kwin_bot.db")
        self.init_database()

    # ------------- базовый коннектор -------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,    # важно для Streamlit
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        # режим WAL для конкуренции чтение/запись
        try:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
        except Exception:
            pass
        return conn

    def init_database(self):
        """Инициализация базы данных (схема + индексы)"""
        with self._connect() as conn:
            c = conn.cursor()

            # Таблица сделок
            c.execute("""
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
                    entry_time TEXT NOT NULL,   -- ISO-8601
                    exit_time TEXT,             -- ISO-8601
                    exit_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Индексы для быстрых выборок
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)")

            # Таблица equity
            c.execute("""
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    timestamp TEXT DEFAULT (datetime('now')) -- ISO-8601
                )
            """)

            # Таблица состояния бота
            c.execute("""
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Таблица конфигурации
            c.execute("""
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY,
                    config_data TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)

            # Таблица логов
            c.execute("""
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    timestamp TEXT DEFAULT (datetime('now'))
                )
            """)

            conn.commit()

    # -------------------- Trades --------------------

    def save_trade(self, trade_data: Dict) -> int:
        """Сохранение новой сделки (возвращает id)."""
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
                    to_iso(trade_data.get("entry_time") or datetime.utcnow()),
                    trade_data.get("status", "open"),
                ),
            )
            conn.commit()
            return c.lastrowid   # <-- возвращаем id

    def add_trade(self, trade_data: Dict) -> int:
        """
        Backward-compat alias.
        Старый код мог вызывать db.add_trade(...). Делаем совместимость,
        проксируя в save_trade(...).
        """
        return self.save_trade(trade_data)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """
        Обновление открытой сделки при выходе + расчёт PnL/RR (нетто, с двойной комиссией).
        Если передан trade_id — обновляем его, иначе берём последнюю открытую.
        """
        with self._connect() as conn:
            c = conn.cursor()

            trade_id = trade_data.get("trade_id")
            if trade_id:
                c.execute("SELECT * FROM trades WHERE id = ? AND status = 'open'", (trade_id,))
            else:
                # Самая последняя открытая по времени входа
                c.execute("SELECT * FROM trades WHERE status = 'open' ORDER BY entry_time DESC LIMIT 1")

            row = c.fetchone()
            if not row:
                print("No open trade found to update")
                return

            cols = [d[0] for d in c.description]
            tr = dict(zip(cols, row))

            entry_price = float(tr["entry_price"])
            quantity = float(tr["quantity"])
            direction = tr["direction"]

            exit_price = float(trade_data.get("exit_price"))
            exit_time = _to_iso(trade_data.get("exit_time") or datetime.utcnow())
            exit_reason = trade_data.get("exit_reason", "exit")
            status = trade_data.get("status", "closed")

            # PnL (нетто)
            gross = (exit_price - entry_price) * quantity if direction == "long" else (entry_price - exit_price) * quantity
            fees = (entry_price + exit_price) * quantity * float(fee_rate)
            pnl_net = gross - fees

            # RR
            rr = None
            if tr.get("stop_loss") is not None:
                risk_per_unit = abs(entry_price - float(tr["stop_loss"]))
                if risk_per_unit > 0:
                    rr = abs(gross) / (risk_per_unit * quantity)

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
                (exit_price, exit_time, exit_reason, float(pnl_net), None if rr is None else float(rr), status, int(tr["id"]))
            )
            conn.commit()
            print(f"Trade updated: id={tr['id']} PnL={pnl_net:.2f} RR={(rr if rr is not None else 0):.2f}")

    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT * FROM trades ORDER BY entry_time DESC LIMIT ?",
                (int(limit),)
            )
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def get_all_trades(self) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT * FROM trades ORDER BY entry_time DESC")
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    def get_trades_by_period(self, days: int) -> List[Dict]:
        """Сделки, у которых entry_time >= now - days."""
        with self._connect() as conn:
            c = conn.cursor()
            start = (datetime.utcnow() - timedelta(days=int(days))).replace(tzinfo=None)
            c.execute(
                """
                SELECT * FROM trades
                 WHERE entry_time >= ?
                 ORDER BY entry_time DESC
                """,
                (_to_iso(start),)
            )
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    # -------------------- Equity --------------------

    def save_equity_snapshot(self, equity: float, ts: Optional[datetime] = None):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                (float(equity), _to_iso(ts or datetime.utcnow()))
            )
            conn.commit()

    def get_equity_history(self, days: int = 30) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            start = (datetime.utcnow() - timedelta(days=int(days))).replace(tzinfo=None)
            c.execute(
                """
                SELECT equity, timestamp FROM equity_history
                 WHERE timestamp >= ?
                 ORDER BY timestamp ASC
                """,
                (_to_iso(start),)
            )
            rows = c.fetchall()
            return [{"equity": float(r[0]), "timestamp": r[1]} for r in rows]

    # Для бэктеста (если требуется списком)
    def get_equity_snapshots(self, days: int = 30) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            start = (datetime.utcnow() - timedelta(days=int(days))).replace(tzinfo=None)
            c.execute(
                "SELECT timestamp as time, equity FROM equity_history WHERE timestamp >= ? ORDER BY timestamp ASC",
                (_to_iso(start),)
            )
            cols = [d[0] for d in c.description]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    # -------------------- State / Config --------------------

    def save_bot_state(self, state: Dict):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                """
                INSERT OR REPLACE INTO bot_state (id, state_data, updated_at)
                VALUES (1, ?, ?)
                """,
                (json.dumps(state, ensure_ascii=False), _to_iso(datetime.utcnow()))
            )
            conn.commit()

    def get_bot_state(self) -> Optional[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT state_data FROM bot_state WHERE id = 1")
            row = c.fetchone()
            return json.loads(row[0]) if row and row[0] else None

    # -------------------- Stats / Logs --------------------

    def get_performance_stats(self, days: int = 30) -> Dict:
        with self._connect() as conn:
            c = conn.cursor()
            start = (datetime.utcnow() - timedelta(days=int(days))).replace(tzinfo=None)
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
                    AVG(CASE 
                        WHEN exit_time IS NOT NULL AND entry_time IS NOT NULL 
                        THEN (julianday(exit_time) - julianday(entry_time)) * 24 
                    END) as avg_hold_hours
                FROM trades 
                WHERE entry_time >= ? AND status = 'closed'
                """,
                (_to_iso(start),)
            )
            row = c.fetchone() or (0,)*8

            total_trades = int(row[0] or 0)
            winning_trades = int(row[1] or 0)
            stats = {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": max(total_trades - winning_trades, 0),
                "win_rate": (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0,
                "avg_pnl": float(row[2] or 0.0),
                "total_pnl": float(row[3] or 0.0),
                "avg_rr": float(row[4] or 0.0),
                "max_win": float(row[5] or 0.0),
                "max_loss": float(row[6] or 0.0),
                "avg_hold_time": float(row[7] or 0.0),
            }
            return stats

    def get_trades_count_today(self) -> int:
        """Количество сделок, открытых сегодня (UTC)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM trades WHERE DATE(entry_time) = DATE('now')")
            return int((c.fetchone() or (0,))[0])

    def get_pnl_today(self) -> float:
        """Суммарный PnL закрытых сделок за сегодня (UTC)."""
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE DATE(exit_time) = DATE('now') AND status = 'closed'"
            )
            return float((c.fetchone() or (0.0,))[0])

    def save_log(self, level: str, message: str, module: str = None):
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO logs (level, message, module, timestamp) VALUES (?, ?, ?, ?)",
                (level, message, module, _to_iso(datetime.utcnow()))
            )
            conn.commit()

    def get_logs(self, limit: int = 100) -> List[Dict]:
        with self._connect() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT level, message, module, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?",
                (int(limit),)
            )
            cols = ["level", "message", "module", "timestamp"]
            return [dict(zip(cols, r)) for r in c.fetchall()]

    # -------------------- Housekeeping --------------------

    def cleanup_old_data(self, days_to_keep: int = 90):
        with self._connect() as conn:
            c = conn.cursor()
            cutoff = (datetime.utcnow() - timedelta(days=int(days_to_keep))).replace(tzinfo=None)
            cutoff_iso = _to_iso(cutoff)

            # Логи
            c.execute("DELETE FROM logs WHERE timestamp < ?", (cutoff_iso,))
            # История equity
            c.execute("DELETE FROM equity_history WHERE timestamp < ?", (cutoff_iso,))
            conn.commit()
