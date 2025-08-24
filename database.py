# database.py
import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta, timezone
import threading


# ========================= Вспомогательные =========================

def _to_iso(ts) -> Optional[str]:
    """Унифицируем хранение времени: всегда ISO (UTC, без микросекунд)."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        if ts.tzinfo is not None:
            ts = ts.astimezone(timezone.utc).replace(tzinfo=None)
        return ts.replace(microsecond=0).isoformat()
    try:
        if isinstance(ts, (int, float)):
            # поддержка unix в мс
            if ts > 1e12:
                ts = ts / 1000.0
            dt = datetime.utcfromtimestamp(float(ts))
            return dt.replace(microsecond=0).isoformat()
        return str(ts)
    except Exception:
        return str(ts)


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Надёжный парсер ISO-строк, возвращает UTC-naive (как мы их храним)."""
    if not s:
        return None
    try:
        # пробуем стандартный fromisoformat
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        try:
            # fallback: попытка unix
            f = float(s)
            if f > 1e12:
                f = f / 1000.0
            return datetime.utcfromtimestamp(f).replace(microsecond=0)
        except Exception:
            return None


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_json(obj: Any) -> Any:
    """Гарантируем json-совместимость для вложенных структур/дат."""
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
        # Базовые тюнинги под многопоточность и стабильность
        try:
            self.conn.execute("PRAGMA journal_mode=WAL;")
            self.conn.execute("PRAGMA synchronous=NORMAL;")
            self.conn.execute("PRAGMA foreign_keys=ON;")
        except Exception:
            pass
        self._lock = threading.RLock()
        self._init_schema()

    def __del__(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

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
        # Индексы
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time  ON trades(exit_time)")
        self.conn.commit()

    # -------------------------- Trades --------------------------
    def save_trade(self, trade: Dict) -> int:
        """Создать запись сделки (открытие). Возвращает ID."""
        with self._lock:
            try:
                # qty / quantity унификация
                qty = trade.get("quantity", None)
                if qty in (None, "", 0) and ("size" in trade):
                    try:
                        qty = float(trade.get("size"))
                    except Exception:
                        qty = trade.get("size")

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
                    _f(qty),
                    _f(trade.get("qty")),  # на всякий случай отдельное поле
                    _f(trade.get("pnl")),
                    _f(trade.get("rr")),
                    _to_iso(trade.get("entry_time") or datetime.utcnow()),
                    _to_iso(trade.get("exit_time")),
                    trade.get("exit_reason"),
                    str(trade.get("status", "open")),
                ))
                self.conn.commit()
                return int(c.lastrowid or 0)
            except Exception:
                self.conn.rollback()
                raise

    def add_trade(self, trade: Dict) -> int:
        return self.save_trade(trade)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """
        Обновляет последнюю открытую сделку (или по id) на закрытие и считает net PnL с комиссиями.
        """
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
                side        = (tr.get("direction") or "").lower()
                exit_price  = _f(trade_data.get("exit_price"), entry_price)

                # === Net PnL с комиссиями ===
                gross = (exit_price - entry_price) * qty if side == "long" else (entry_price - exit_price) * qty
                fee_in  = entry_price * qty * fee_rate
                fee_out = exit_price * qty * fee_rate
                net_pnl = gross - (fee_in + fee_out)

                rr = None
                if stop_loss not in (None, 0.0):
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

    # -------------------------- Aggregates for UI --------------------------
    def get_trades_count_today(self) -> int:
        """Количество сделок, открытых сегодня (UTC) — для дашборда."""
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            c.execute(
                "SELECT COUNT(1) FROM trades WHERE entry_time >= ?",
                (_to_iso(start),),
            )
            row = c.fetchone()
            return int(row[0] if row and row[0] is not None else 0)

    def get_pnl_today(self) -> float:
        """Сумма PnL закрытых сделок за сегодня (UTC)."""
        with self._lock:
            c = self.conn.cursor()
            start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            c.execute(
                "SELECT COALESCE(SUM(pnl), 0.0) FROM trades WHERE status = 'closed' AND exit_time >= ?",
                (_to_iso(start),),
            )
            row = c.fetchone()
            return float(row[0] if row and row[0] is not None else 0.0)

    def get_performance_stats(self, days: int = 30) -> Dict[str, Any]:
        """
        Сводка для UI: win_rate, avg_rr, avg_hold_time (часы).
        Берём сделки за последние `days` дней; учитываем только закрытые для метрик,
        требующих exit_time (hold time).
        """
        with self._lock:
            cutoff = datetime.utcnow() - timedelta(days=int(days))
            cutoff_iso = _to_iso(cutoff)

            c = self.conn.cursor()
            # все сделки за период (для total)
            c.execute(
                "SELECT direction, status, rr, pnl, entry_time, exit_time "
                "FROM trades WHERE entry_time >= ? ORDER BY entry_time ASC",
                (cutoff_iso,)
            )
            rows = [dict(r) for r in c.fetchall()]

        total = len(rows)
        closed = [r for r in rows if str(r.get("status", "")).lower() == "closed"]
        wins = [r for r in closed if _f(r.get("pnl"), 0.0) > 0.0]
        losses = [r for r in closed if _f(r.get("pnl"), 0.0) < 0.0]

        # win rate
        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0

        # avg rr
        rr_vals = [_f(r.get("rr")) for r in closed if r.get("rr") not in (None, "")]
        avg_rr = (sum(rr_vals) / len(rr_vals)) if rr_vals else 0.0

        # avg hold time (часы)
        hold_hours = []
        for r in closed:
            et = _parse_iso(r.get("entry_time"))
            xt = _parse_iso(r.get("exit_time"))
            if et and xt and xt >= et:
                hold_hours.append((xt - et).total_seconds() / 3600.0)
        avg_hold = (sum(hold_hours) / len(hold_hours)) if hold_hours else 0.0

        return {
            "total": total,
            "closed": len(closed),
            "win_rate": round(win_rate, 2),
            "avg_rr": round(avg_rr, 3),
            "avg_hold_time": round(avg_hold, 3),
        }

    # -------------------------- Equity --------------------------
    def save_equity_snapshot(self, equity: float):
        """Асинхронный write, чтобы не блокировать основной поток."""
        def _worker(eq):
            try:
                with self._lock:
                    c = self.conn.cursor()
                    c.execute(
                        "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                        (_f(eq, 0.0), _to_iso(datetime.utcnow()))
                    )
                    self.conn.commit()
            except Exception:
                # не шумим в фоне
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
                (str(level), str(message), module, _to_iso(datetime.utcnow())),
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
            c.execute(
                "DELETE FROM logs WHERE id NOT IN (SELECT id FROM logs ORDER BY id DESC LIMIT ?)",
                (int(keep),)
            )
            self.conn.commit()

    # --------------------- Bot State / Config ---------------------
    def get_bot_state(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT state_data FROM bot_state WHERE id = 1")
            row = c.fetchone()
            try:
                return json.loads(row[0]) if row else {}
            except Exception:
                return {}

    def save_bot_state(self, state: Dict[str, Any]):
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO bot_state (id, state_data, updated_at) VALUES (1, ?, ?)",
                (json.dumps(_normalize_json(state), ensure_ascii=False), _to_iso(datetime.utcnow()))
            )
            self.conn.commit()

    def get_config(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT config_data FROM config WHERE id = 1")
            row = c.fetchone()
            try:
                return json.loads(row[0]) if row else {}
            except Exception:
                return {}

    def save_config(self, cfg: Dict[str, Any]):
        with self._lock:
            c = self.conn.cursor()
            c.execute(
                "INSERT OR REPLACE INTO config (id, config_data, updated_at) VALUES (1, ?, ?)",
                (json.dumps(_normalize_json(cfg), ensure_ascii=False), _to_iso(datetime.utcnow()))
            )
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
            with self._lock:
                c = self.conn.cursor()
                c.execute(
                    "INSERT INTO equity_history (equity, timestamp) VALUES (?, ?)",
                    (_f(e.get("equity")), _to_iso(e.get("timestamp") or datetime.utcnow()))
                )
                self.conn.commit()
        self.save_bot_state(dump.get("state", {}))
        self.save_config(dump.get("config", {}))
        for lg in dump.get("logs", []):
            self.save_log(lg.get("level", "info"), lg.get("message", ""), lg.get("module"))

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
        """Полностью очищает БД перед новым бэктестом."""
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
