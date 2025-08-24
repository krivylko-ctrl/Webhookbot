"""
📈 Модуль аналитики и статистики торгов KWIN Strategy
Предоставляет детальную аналитику сделок, winrate, PnL, ROI и другие метрики
"""
from typing import Dict, List, Optional
from datetime import datetime, timedelta, timezone
import sqlite3


# ---------- утилиты времени (совместимо с database.py) ----------
def _to_iso(dt: datetime) -> str:
    # всегда naive-UTC без микросекунд
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.replace(microsecond=0).isoformat()


def _as_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


class TradingAnalytics:
    """📊 Аналитика торгов с полной статистикой как в TradingView"""

    def __init__(self, db_path: str = "kwin_bot.db"):
        self.db_path = db_path

    # ---------- RAW ----------
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def get_trades_data(self, days_back: int = 30) -> List[Dict]:
        """
        Получение закрытых/остановленных сделок за период (UTC),
        совместимо со схемой trades из database.py
        """
        try:
            conn = self._connect()
            cur = conn.cursor()

            start_iso = _to_iso(datetime.utcnow() - timedelta(days=int(days_back)))

            cur.execute(
                """
                SELECT *
                  FROM trades
                 WHERE entry_time >= ?
                   AND status IN ('closed','stopped','stop','tp','sl')
                 ORDER BY entry_time DESC
                """,
                (start_iso,),
            )
            rows = cur.fetchall()
            trades: List[Dict] = []
            for r in rows:
                t = dict(r)

                # нормализация чисел
                for key in ("pnl", "entry_price", "exit_price", "stop_loss", "take_profit"):
                    if key in t:
                        t[key] = _as_float(t.get(key))

                # qty/quantity унификация
                qty = t.get("quantity", None)
                if qty in (None, "", 0):
                    qty = t.get("qty", 0)
                t["quantity"] = _as_float(qty)

                # rr поле: в БД оно называется 'rr'
                rr = t.get("rr", None)
                if rr in (None, "") and "risk_reward" in t:
                    rr = t.get("risk_reward")
                t["rr"] = _as_float(rr)

                trades.append(t)

            conn.close()
            return trades
        except Exception as e:
            print(f"[analytics] Error getting trades data: {e}")
            return []

    def _get_equity_history(self, days_back: int = 30) -> List[Dict]:
        try:
            conn = self._connect()
            cur = conn.cursor()
            start_iso = _to_iso(datetime.utcnow() - timedelta(days=int(days_back)))
            cur.execute(
                """
                SELECT equity, timestamp
                  FROM equity_history
                 WHERE timestamp >= ?
                 ORDER BY timestamp ASC
                """,
                (start_iso,),
            )
            rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[analytics] Error reading equity_history: {e}")
            return []

    # ---------- METRICS ----------
    def calculate_winrate(self, trades: List[Dict]) -> Dict[str, float]:
        """📊 Winrate total/long/short"""
        if not trades:
            return {"total": 0.0, "long": 0.0, "short": 0.0}

        def _wr(sub: List[Dict]) -> float:
            if not sub:
                return 0.0
            wins = sum(1 for t in sub if _as_float(t.get("pnl")) > 0.0)
            return round(100.0 * wins / len(sub), 2)

        long_trades = [t for t in trades if (t.get("direction") == "long")]
        short_trades = [t for t in trades if (t.get("direction") == "short")]

        return {
            "total": _wr(trades),
            "long": _wr(long_trades),
            "short": _wr(short_trades),
        }

    def calculate_pnl_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """💰 PnL / Avg win/loss / Profit Factor"""
        if not trades:
            return {"total_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "profit_factor": 0.0}

        pnls = [_as_float(t.get("pnl")) for t in trades]
        total_pnl = sum(pnls)

        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        gross_profit = sum(wins) if wins else 0.0
        gross_loss = abs(sum(losses)) if losses else 0.0
        avg_win = (gross_profit / len(wins)) if wins else 0.0
        avg_loss = (gross_loss / len(losses)) if losses else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

        return {
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
        }

    def calculate_risk_reward_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """⚖️ Risk/Reward (используем поле 'rr' из БД)"""
        rr_vals = [_as_float(t.get("rr")) for t in trades if t.get("rr") not in (None, "", 0)]
        if not rr_vals:
            return {"avg_rr": 0.0, "max_rr": 0.0, "min_rr": 0.0}
        return {
            "avg_rr": round(sum(rr_vals) / len(rr_vals), 2),
            "max_rr": round(max(rr_vals), 2),
            "min_rr": round(min(rr_vals), 2),
        }

    def calculate_drawdown_metrics(self, equity_history: List[Dict]) -> Dict[str, float]:
        """📉 Max DD и текущая DD по equity_history"""
        if not equity_history:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0}

        eq = [_as_float(e.get("equity")) for e in equity_history if e.get("equity") is not None]
        if not eq:
            return {"max_drawdown": 0.0, "current_drawdown": 0.0}

        peak = eq[0]
        max_dd = 0.0
        for v in eq:
            if v > peak:
                peak = v
            dd = (v - peak) / (peak if peak != 0 else 1.0) * 100.0
            if dd < max_dd:
                max_dd = dd

        current_peak = max(eq)  # упрощённый текущий пик
        curr_dd = (eq[-1] - current_peak) / (current_peak if current_peak != 0 else 1.0) * 100.0

        return {"max_drawdown": round(abs(max_dd), 2), "current_drawdown": round(abs(curr_dd), 2)}

    def calculate_roi_metrics(self, trades: List[Dict], initial_balance: float = 1000.0) -> Dict[str, float]:
        """💹 ROI total/monthly/daily (грубая оценка по PnL сделок)"""
        if not trades:
            return {"total_roi": 0.0, "monthly_roi": 0.0, "daily_roi": 0.0}

        total_pnl = sum(_as_float(t.get("pnl")) for t in trades)
        total_roi = (total_pnl / float(initial_balance)) * 100.0 if initial_balance > 0 else 0.0

        # период по exit_time (если есть), иначе по entry_time
        times: List[datetime] = []
        for t in trades:
            ts = t.get("exit_time") or t.get("entry_time")
            if ts:
                try:
                    if isinstance(ts, str):
                        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:
                    ts = None
            if isinstance(ts, datetime):
                times.append(ts)
        if times:
            days = max((max(times) - min(times)).days, 1)
            monthly_roi = total_roi / days * 30.0
            daily_roi = total_roi / days
        else:
            monthly_roi = 0.0
            daily_roi = 0.0

        return {
            "total_roi": round(total_roi, 2),
            "monthly_roi": round(monthly_roi, 2),
            "daily_roi": round(daily_roi, 2),
        }

    # ---------- PUBLIC ----------
    def get_comprehensive_stats(self, days_back: int = 30) -> Dict:
        """🎯 Комплексная сводка за период"""
        try:
            trades = self.get_trades_data(days_back=days_back)
            equity_history = self._get_equity_history(days_back=days_back)

            return {
                "period_days": int(days_back),
                "total_trades": len(trades),
                "winrate": self.calculate_winrate(trades),
                "pnl": self.calculate_pnl_metrics(trades),
                "risk_reward": self.calculate_risk_reward_metrics(trades),
                "drawdown": self.calculate_drawdown_metrics(equity_history),
                "roi": self.calculate_roi_metrics(trades),
                "updated_at": _to_iso(datetime.utcnow()),
            }
        except Exception as e:
            print(f"[analytics] Error in comprehensive stats: {e}")
            return {}


class TrailingLogger:
    """4️⃣ Логгер движения трейлинга (совместим с current DB)"""

    def __init__(self, db_path: str = "kwin_bot.db"):
        self.db_path = db_path
        self.init_trailing_table()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_trailing_table(self):
        """Создание таблицы для логов трейлинга"""
        try:
            conn = self._connect()
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS trailing_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    old_sl REAL,
                    new_sl REAL,
                    current_price REAL,
                    trigger_type TEXT,
                    arm_status TEXT,
                    lookback_value REAL,
                    buffer_ticks INTEGER,
                    trail_distance REAL,
                    unrealized_pnl REAL
                )
                """
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[analytics] Error creating trailing table: {e}")

    def log_trail_movement(
        self,
        position: Dict,
        old_sl: float,
        new_sl: float,
        current_price: float,
        trigger_type: str,
        **kwargs,
    ):
        """Логирование движения трейла"""
        try:
            conn = self._connect()
            cur = conn.cursor()

            entry_price = _as_float(position.get("entry_price"))
            direction = position.get("direction", "long")
            qty = _as_float(position.get("size") or position.get("quantity"))

            if direction == "long":
                unrealized_pnl = (current_price - entry_price) * qty
            else:
                unrealized_pnl = (entry_price - current_price) * qty

            cur.execute(
                """
                INSERT INTO trailing_logs (
                    symbol, direction, entry_price, old_sl, new_sl, current_price,
                    trigger_type, arm_status, lookback_value, buffer_ticks,
                    trail_distance, unrealized_pnl
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position.get("symbol", "ETHUSDT"),
                    direction,
                    entry_price,
                    _as_float(old_sl),
                    _as_float(new_sl),
                    _as_float(current_price),
                    str(trigger_type),
                    "armed" if position.get("armed", False) else "disarmed",
                    _as_float(kwargs.get("lookback_value", 0)),
                    int(kwargs.get("buffer_ticks", 0) or 0),
                    abs(_as_float(new_sl) - _as_float(current_price)),
                    _as_float(unrealized_pnl),
                ),
            )

            conn.commit()
            conn.close()
            print(
                f"Trail logged: {trigger_type} | SL: {old_sl:.6f} → {new_sl:.6f} | Price: {current_price:.6f}"
            )
        except Exception as e:
            print(f"[analytics] Error logging trail movement: {e}")

    def get_trailing_history(self, hours_back: int = 24) -> List[Dict]:
        """Получение истории трейлинга"""
        try:
            conn = self._connect()
            cur = conn.cursor()
            start_iso = _to_iso(datetime.utcnow() - timedelta(hours=int(hours_back)))

            cur.execute(
                """
                SELECT * FROM trailing_logs
                 WHERE timestamp >= ?
                 ORDER BY timestamp DESC
                """,
                (start_iso,),
            )
            rows = cur.fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as e:
            print(f"[analytics] Error getting trailing history: {e}")
            return []
