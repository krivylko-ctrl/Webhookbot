# database.py
import sqlite3
import json
from typing import Dict, List, Optional, Any, Iterable, Tuple
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
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except Exception:
        try:
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
        # миграции безболезненно добавят колонки/индексы, если их ещё нет
        try:
            self.migrate()
        except Exception:
            pass

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

    # -------------------------- Внутренние утилиты --------------------------
    def _has_column(self, table: str, column: str) -> bool:
        try:
            cur = self.conn.execute(f"PRAGMA table_info({table})")
            cols = [r[1] for r in cur.fetchall()]
            return column in cols
        except Exception:
            return False

    # -------------------------- Схема --------------------------
    def _init_schema(self) -> None:
        c = self.conn.cursor()
        # Базовые таблицы
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
        # Опциональные таблицы (candles / ws_health)
        c.execute("""
            CREATE TABLE IF NOT EXISTS candles (
                symbol   TEXT NOT NULL,
                tf       TEXT NOT NULL,
                ts_ms    INTEGER NOT NULL,
                open     REAL NOT NULL,
                high     REAL NOT NULL,
                low      REAL NOT NULL,
                close    REAL NOT NULL,
                volume   REAL NOT NULL,
                PRIMARY KEY (symbol, tf, ts_ms)
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS ws_health (
                id               INTEGER PRIMARY KEY CHECK (id=1),
                last_pong_at     TEXT,
                last_1m_close_ms INTEGER,
                last_15m_close_ms INTEGER,
                last_60m_close_ms INTEGER,
                updated_at       TEXT
            )
        """)
        # Индексы
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_trades_exit_time  ON trades(exit_time)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts ON candles(symbol, tf, ts_ms)")
        self.conn.commit()

    # -------------------------- Миграции --------------------------
    def migrate(self):
        """Безопасные ALTER'ы: добавляют расширения аудита, если их нет."""
        cur = self.conn.cursor()

        def _alter(sql: str):
            try:
                cur.execute(sql)
                self.conn.commit()
            except Exception:
                self.conn.rollback()

        # trades расширения
        _alter("ALTER TABLE trades ADD COLUMN signal_source TEXT")
        _alter("ALTER TABLE trades ADD COLUMN entry_tf TEXT")
        _alter("ALTER TABLE trades ADD COLUMN entry_bar_ts_ms INTEGER")
        _alter("ALTER TABLE trades ADD COLUMN entry_1m_ts_ms INTEGER")
        _alter("ALTER TABLE trades ADD COLUMN swing_id INTEGER")
        _alter("ALTER TABLE trades ADD COLUMN sl_calc REAL")
        _alter("ALTER TABLE trades ADD COLUMN sl_api REAL")
        _alter("ALTER TABLE trades ADD COLUMN fee_rate REAL")
        _alter("ALTER TABLE trades ADD COLUMN fee_in REAL")
        _alter("ALTER TABLE trades ADD COLUMN fee_out REAL")
        _alter("ALTER TABLE trades ADD COLUMN risk_pct REAL")
        _alter("ALTER TABLE trades ADD COLUMN rr_target REAL")
        _alter("ALTER TABLE trades ADD COLUMN arm_rr REAL")
        _alter("ALTER TABLE trades ADD COLUMN trailing_perc REAL")
        _alter("ALTER TABLE trades ADD COLUMN trailing_offset_perc REAL")
        _alter("ALTER TABLE trades ADD COLUMN trail_activated INTEGER")  # 0/1
        _alter("ALTER TABLE trades ADD COLUMN trail_activated_at TEXT")
        _alter("ALTER TABLE trades ADD COLUMN trail_last_update_at TEXT")
        _alter("ALTER TABLE trades ADD COLUMN exchange TEXT")
        _alter("ALTER TABLE trades ADD COLUMN account TEXT")
        _alter("ALTER TABLE trades ADD COLUMN strategy TEXT")

        # Индексы под отчёты
        _alter("CREATE INDEX IF NOT EXISTS idx_trades_symbol_time ON trades(symbol, entry_time)")
        _alter("CREATE INDEX IF NOT EXISTS idx_trades_swing_id ON trades(swing_id)")
        _alter("CREATE INDEX IF NOT EXISTS idx_trades_source_tf ON trades(signal_source, entry_tf)")

    # -------------------------- Свечи (опционально) --------------------------
    def upsert_candle(self, symbol: str, tf: str, ts_ms: int,
                      open_: float, high: float, low: float, close: float, volume: float):
        """INSERT OR REPLACE закрытой свечи. Необязательно к использованию (есть CSV)."""
        with self._lock:
            c = self.conn.cursor()
            c.execute("""
                INSERT OR REPLACE INTO candles (symbol, tf, ts_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (str(symbol), str(tf), int(ts_ms), _f(open_), _f(high), _f(low), _f(close), _f(volume)))
            self.conn.commit()

    def bulk_upsert_candles(self, rows: Iterable[Tuple[str, str, int, float, float, float, float, float]]):
        """Пакетная загрузка свечей: список кортежей (symbol, tf, ts_ms, o,h,l,c,v)."""
        with self._lock:
            c = self.conn.cursor()
            c.executemany("""
                INSERT OR REPLACE INTO candles (symbol, tf, ts_ms, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [(str(a), str(b), int(c0), _f(d), _f(e), _f(f0), _f(g), _f(h0)) for (a,b,c0,d,e,f0,g,h0) in rows])
            self.conn.commit()

    # -------------------------- WS Health (опционально) --------------------------
    def update_ws_health(self, last_pong_at: Optional[datetime] = None,
                         last_1m_close_ms: Optional[int] = None,
                         last_15m_close_ms: Optional[int] = None,
                         last_60m_close_ms: Optional[int] = None):
        """Поддержка мониторинга соединения/закрытых баров."""
        with self._lock:
            c = self.conn.cursor()
            # получаем текущую запись (единственная, id=1)
            c.execute("SELECT id FROM ws_health WHERE id=1")
            row = c.fetchone()
            if not row:
                c.execute("INSERT INTO ws_health (id) VALUES (1)")
                self.conn.commit()
            # динамический апдейт
            fields = []
            vals = []
            if last_pong_at is not None:
                fields.append("last_pong_at=?")
                vals.append(_to_iso(last_pong_at))
            if last_1m_close_ms is not None:
                fields.append("last_1m_close_ms=?")
                vals.append(int(last_1m_close_ms))
            if last_15m_close_ms is not None:
                fields.append("last_15m_close_ms=?")
                vals.append(int(last_15m_close_ms))
            if last_60m_close_ms is not None:
                fields.append("last_60m_close_ms=?")
                vals.append(int(last_60m_close_ms))
            fields.append("updated_at=?")
            vals.append(_to_iso(datetime.utcnow()))
            sql = f"UPDATE ws_health SET {', '.join(fields)} WHERE id=1"
            c.execute(sql, tuple(vals))
            self.conn.commit()

    def get_ws_health(self) -> Dict[str, Any]:
        with self._lock:
            c = self.conn.cursor()
            c.execute("SELECT * FROM ws_health WHERE id=1")
            row = c.fetchone()
            return dict(row) if row else {}

    # -------------------------- Trades --------------------------
    def save_trade(self, trade: Dict) -> int:
        """Создать запись сделки (открытие). Возвращает ID. Допполя пишем отдельным UPDATE'ом."""
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
                trade_id = int(c.lastrowid or 0)

                # --- Доп. поля аудита: UPDATE по месту, только если колонки существуют и значения переданы ---
                update_fields = []
                update_vals = []

                def _maybe(col, key, transform=lambda x: x):
                    if self._has_column("trades", col) and (key in trade) and (trade.get(key) is not None):
                        update_fields.append(f"{col}=?")
                        update_vals.append(transform(trade.get(key)))

                _maybe("signal_source", "signal_source", str)
                _maybe("entry_tf", "entry_tf", str)
                _maybe("entry_bar_ts_ms", "entry_bar_ts_ms", int)
                _maybe("entry_1m_ts_ms", "entry_1m_ts_ms", int)
                _maybe("swing_id", "swing_id", int)
                _maybe("sl_calc", "sl_calc", _f)
                _maybe("sl_api", "sl_api", _f)
                _maybe("fee_rate", "fee_rate", _f)
                _maybe("risk_pct", "risk_pct", _f)
                _maybe("rr_target", "rr_target", _f)
                _maybe("arm_rr", "arm_rr", _f)
                _maybe("trailing_perc", "trailing_perc", _f)
                _maybe("trailing_offset_perc", "trailing_offset_perc", _f)
                _maybe("trail_activated", "trail_activated", lambda x: int(bool(x)))
                _maybe("trail_activated_at", "trail_activated_at", _to_iso)
                _maybe("trail_last_update_at", "trail_last_update_at", _to_iso)
                _maybe("exchange", "exchange", str)
                _maybe("account", "account", str)
                _maybe("strategy", "strategy", str)

                if update_fields:
                    sql = f"UPDATE trades SET {', '.join(update_fields)} WHERE id=?"
                    update_vals.append(trade_id)
                    c.execute(sql, tuple(update_vals))

                self.conn.commit()
                return trade_id
            except Exception:
                self.conn.rollback()
                raise

    def add_trade(self, trade: Dict) -> int:
        return self.save_trade(trade)

    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """
        Обновляет последнюю открытую сделку (или по id) на закрытие и считает net PnL с комиссиями.
        Также сохраняет fee_in/fee_out/fee_rate при наличии колонок.
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

                # Базовый апдейт
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

                # Доп. поля комиссий, если есть
                cols = []
                vals = []
                if self._has_column("trades", "fee_rate"):
                    cols.append("fee_rate=?"); vals.append(_f(fee_rate))
                if self._has_column("trades", "fee_in"):
                    cols.append("fee_in=?"); vals.append(_f(fee_in))
                if self._has_column("trades", "fee_out"):
                    cols.append("fee_out=?"); vals.append(_f(fee_out))
                if cols:
                    sql = f"UPDATE trades SET {', '.join(cols)} WHERE id=?"
                    vals.append(tr["id"])
                    c.execute(sql, tuple(vals))

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

        win_rate = (len(wins) / len(closed) * 100.0) if closed else 0.0

        rr_vals = [_f(r.get("rr")) for r in closed if r.get("rr") not in (None, "")]
        avg_rr = (sum(rr_vals) / len(rr_vals)) if rr_vals else 0.0

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
            # Свечи не включаю по умолчанию, чтобы не раздувать даump; добавим по запросу
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
                DROP TABLE IF EXISTS candles;
                DROP TABLE IF EXISTS ws_health;
            """)
            self.conn.commit()
        self._init_schema()
        try:
            self.migrate()
        except Exception:
            pass

    # --------------------- Reset для Backtest ---------------------
    def reset_for_backtest(self):
        """Полностью очищает БД перед новым бэктестом (без дропа таблиц)."""
        with self._lock:
            c = self.conn.cursor()
            c.executescript("""
                DELETE FROM trades;
                DELETE FROM equity_history;
                DELETE FROM bot_state;
                DELETE FROM config;
                DELETE FROM logs;
                DELETE FROM candles;
                DELETE FROM ws_health;
            """)
            self.conn.commit()
