"""
📈 Модуль аналитики и статистики торгов KWIN Strategy
Предоставляет детальную аналитику сделок, winrate, PnL, ROI и другие метрики
"""
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import sqlite3
import json

try:
    import pandas as pd
    import numpy as np
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

class TradingAnalytics:
    """📊 Аналитика торгов с полной статистикой как в TradingView"""
    
    def __init__(self, db_path: str = "kwin_bot.db"):
        self.db_path = db_path
    
    def get_trades_data(self, days_back: int = 30) -> List[Dict]:
        """Получение данных сделок за период"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Фильтр по датам
            start_date = datetime.now() - timedelta(days=days_back)
            
            query = """
            SELECT * FROM trades 
            WHERE entry_time >= ? AND status IN ('closed', 'stopped')
            ORDER BY entry_time DESC
            """
            
            cursor.execute(query, [start_date])
            rows = cursor.fetchall()
            
            # Получаем имена колонок
            columns = [description[0] for description in cursor.description]
            
            # Преобразуем в список словарей
            trades = []
            for row in rows:
                trade = dict(zip(columns, row))
                # Конвертируем типы
                if trade.get('pnl'):
                    trade['pnl'] = float(trade['pnl'])
                if trade.get('entry_price'):
                    trade['entry_price'] = float(trade['entry_price'])
                if trade.get('exit_price'):
                    trade['exit_price'] = float(trade['exit_price'])
                if trade.get('quantity'):
                    trade['quantity'] = float(trade['quantity'])
                trades.append(trade)
            
            conn.close()
            return trades
            
        except Exception as e:
            print(f"Error getting trades data: {e}")
            return []
    
    def calculate_winrate(self, trades: List[Dict]) -> Dict[str, float]:
        """📊 Расчет winrate по типам сделок"""
        if not trades:
            return {"total": 0, "long": 0, "short": 0}
        
        # Общий winrate
        total_trades = len(trades)
        winning_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_winrate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Winrate по направлениям
        long_trades = [t for t in trades if t.get('direction') == 'long']
        short_trades = [t for t in trades if t.get('direction') == 'short']
        
        long_winrate = 0
        short_winrate = 0
        
        if long_trades:
            long_wins = len([t for t in long_trades if t.get('pnl', 0) > 0])
            long_winrate = (long_wins / len(long_trades)) * 100
        
        if short_trades:
            short_wins = len([t for t in short_trades if t.get('pnl', 0) > 0])
            short_winrate = (short_wins / len(short_trades)) * 100
        
        return {
            "total": round(total_winrate, 2),
            "long": round(long_winrate, 2),
            "short": round(short_winrate, 2)
        }
    
    def calculate_pnl_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """💰 Расчет PnL метрик"""
        if not trades:
            return {"total_pnl": 0, "avg_win": 0, "avg_loss": 0, "profit_factor": 0}
        
        # Общий PnL
        total_pnl = sum([t.get('pnl', 0) for t in trades])
        
        # Прибыльные и убыточные сделки
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        
        # Средние значения
        avg_win = sum([t['pnl'] for t in winning_trades]) / len(winning_trades) if winning_trades else 0
        avg_loss = abs(sum([t['pnl'] for t in losing_trades]) / len(losing_trades)) if losing_trades else 0
        
        # Profit Factor
        gross_profit = sum([t['pnl'] for t in winning_trades]) if winning_trades else 0
        gross_loss = abs(sum([t['pnl'] for t in losing_trades])) if losing_trades else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return {
            "total_pnl": round(total_pnl, 2),
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2)
        }
    
    def calculate_risk_reward_metrics(self, trades: List[Dict]) -> Dict[str, float]:
        """⚖️ Расчет Risk/Reward метрик"""
        if not trades:
            return {"avg_rr": 0, "max_rr": 0, "min_rr": 0}
        
        # Фильтруем сделки с RR данными
        rr_trades = [t for t in trades if t.get('risk_reward') is not None]
        
        if not rr_trades:
            return {"avg_rr": 0, "max_rr": 0, "min_rr": 0}
        
        rr_values = [float(t['risk_reward']) for t in rr_trades]
        avg_rr = sum(rr_values) / len(rr_values)
        max_rr = max(rr_values)
        min_rr = min(rr_values)
        
        return {
            "avg_rr": round(avg_rr, 2),
            "max_rr": round(max_rr, 2),
            "min_rr": round(min_rr, 2)
        }
    
    def calculate_drawdown_metrics(self, equity_history: List[Dict]) -> Dict[str, float]:
        """📉 Расчет просадки"""
        if not equity_history:
            return {"max_drawdown": 0, "current_drawdown": 0}
        
        # Простой расчет без pandas
        equity_values = [float(item.get('equity', 0)) for item in equity_history]
        
        if not equity_values:
            return {"max_drawdown": 0, "current_drawdown": 0}
        
        # Расчет просадки
        max_drawdown = 0
        current_peak = equity_values[0]
        
        for equity in equity_values:
            if equity > current_peak:
                current_peak = equity
            else:
                drawdown = ((equity - current_peak) / current_peak) * 100
                max_drawdown = min(max_drawdown, drawdown)
        
        # Текущая просадка
        current_drawdown = ((equity_values[-1] - current_peak) / current_peak) * 100
        
        return {
            "max_drawdown": round(abs(max_drawdown), 2),
            "current_drawdown": round(abs(current_drawdown), 2)
        }
    
    def calculate_roi_metrics(self, trades: List[Dict], initial_balance: float = 1000) -> Dict[str, float]:
        """💹 Расчет ROI метрик"""
        if not trades:
            return {"total_roi": 0, "monthly_roi": 0, "daily_roi": 0}
        
        # Общий PnL
        total_pnl = sum([t.get('pnl', 0) for t in trades])
        total_roi = (total_pnl / initial_balance) * 100
        
        # Временные метрики
        if trades:
            # Получаем даты
            exit_times = [t.get('exit_time') for t in trades if t.get('exit_time')]
            if exit_times:
                try:
                    dates = [datetime.fromisoformat(t.replace('Z', '+00:00')) if isinstance(t, str) else t for t in exit_times]
                    period_days = (max(dates) - min(dates)).days
                    monthly_roi = (total_roi / period_days) * 30 if period_days > 0 else 0
                    daily_roi = total_roi / period_days if period_days > 0 else 0
                except:
                    monthly_roi = 0
                    daily_roi = 0
            else:
                monthly_roi = 0
                daily_roi = 0
        else:
            monthly_roi = 0
            daily_roi = 0
        
        return {
            "total_roi": round(total_roi, 2),
            "monthly_roi": round(monthly_roi, 2),
            "daily_roi": round(daily_roi, 2)
        }
    
    def get_comprehensive_stats(self, days_back: int = 30) -> Dict:
        """🎯 Получение всех статистик за период"""
        try:
            # Получаем данные сделок
            trades = self.get_trades_data(days_back)
            
            # Получаем историю equity
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            equity_query = """
            SELECT * FROM equity_history 
            WHERE timestamp >= ? 
            ORDER BY timestamp
            """
            start_date = datetime.now() - timedelta(days=days_back)
            cursor.execute(equity_query, [start_date])
            equity_rows = cursor.fetchall()
            
            # Получаем имена колонок для equity
            equity_columns = [description[0] for description in cursor.description]
            equity_history = [dict(zip(equity_columns, row)) for row in equity_rows]
            
            conn.close()
            
            # Расчет всех метрик
            stats = {
                "period_days": days_back,
                "total_trades": len(trades),
                "winrate": self.calculate_winrate(trades),
                "pnl": self.calculate_pnl_metrics(trades),
                "risk_reward": self.calculate_risk_reward_metrics(trades),
                "drawdown": self.calculate_drawdown_metrics(equity_history),
                "roi": self.calculate_roi_metrics(trades),
                "updated_at": datetime.now().isoformat()
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating comprehensive stats: {e}")
            return {}
    
    def get_monthly_performance(self) -> List[Dict]:
        """📅 Месячная производительность"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                strftime('%Y-%m', entry_time) as month,
                COUNT(*) as trades,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl,
                AVG(pnl) as avg_pnl
            FROM trades 
            WHERE status IN ('closed', 'stopped')
            GROUP BY month
            ORDER BY month DESC
            LIMIT 12
            """
            
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Получаем имена колонок
            columns = [description[0] for description in cursor.description]
            result = []
            
            for row in rows:
                record = dict(zip(columns, row))
                # Добавляем winrate
                record['winrate'] = (record['wins'] / record['trades']) * 100 if record['trades'] > 0 else 0
                result.append(record)
            
            conn.close()
            return result
            
        except Exception as e:
            print(f"Error getting monthly performance: {e}")
            return []
    
    def get_sfp_pattern_stats(self) -> Dict:
        """🔍 Статистика SFP паттернов"""
        try:
            trades = self.get_trades_data(180)  # 6 месяцев
            
            if not trades:
                return {"bull_sfp": {}, "bear_sfp": {}}
            
            # Разделяем по типам
            bull_trades = [t for t in trades if t.get('direction') == 'long']
            bear_trades = [t for t in trades if t.get('direction') == 'short']
            
            def get_pattern_stats(trades_list, pattern_name):
                if not trades_list:
                    return {"trades": 0, "winrate": 0, "avg_pnl": 0}
                
                winning_trades = [t for t in trades_list if t.get('pnl', 0) > 0]
                total_pnl = sum([t.get('pnl', 0) for t in trades_list])
                avg_pnl = total_pnl / len(trades_list)
                
                return {
                    "trades": len(trades_list),
                    "winrate": round((len(winning_trades) / len(trades_list)) * 100, 2),
                    "avg_pnl": round(avg_pnl, 2),
                    "total_pnl": round(total_pnl, 2)
                }
            
            return {
                "bull_sfp": get_pattern_stats(bull_trades, "Bull SFP"),
                "bear_sfp": get_pattern_stats(bear_trades, "Bear SFP")
            }
            
        except Exception as e:
            print(f"Error getting SFP pattern stats: {e}")
            return {"bull_sfp": {}, "bear_sfp": {}}

class TrailingLogger:
    """4️⃣ Логгер движения трейлинга"""
    
    def __init__(self, db_path: str = "kwin_bot.db"):
        self.db_path = db_path
        self.init_trailing_table()
    
    def init_trailing_table(self):
        """Создание таблицы для логов трейлинга"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
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
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error creating trailing table: {e}")
    
    def log_trail_movement(self, position: Dict, old_sl: float, new_sl: float, 
                          current_price: float, trigger_type: str, **kwargs):
        """Логирование движения трейла"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Расчет нереализованного PnL
            entry_price = position.get('entry_price', 0)
            direction = position.get('direction', 'long')
            quantity = position.get('size', 0)
            
            if direction == 'long':
                unrealized_pnl = (current_price - entry_price) * quantity
            else:
                unrealized_pnl = (entry_price - current_price) * quantity
            
            cursor.execute("""
            INSERT INTO trailing_logs (
                symbol, direction, entry_price, old_sl, new_sl, current_price,
                trigger_type, arm_status, lookback_value, buffer_ticks,
                trail_distance, unrealized_pnl
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                position.get('symbol', 'ETHUSDT'),
                direction,
                entry_price,
                old_sl,
                new_sl,
                current_price,
                trigger_type,
                'armed' if position.get('armed', False) else 'disarmed',
                kwargs.get('lookback_value', 0),
                kwargs.get('buffer_ticks', 0),
                abs(new_sl - current_price),
                unrealized_pnl
            ))
            
            conn.commit()
            conn.close()
            
            print(f"Trail logged: {trigger_type} | SL: {old_sl:.2f} → {new_sl:.2f} | Price: {current_price:.2f}")
            
        except Exception as e:
            print(f"Error logging trail movement: {e}")
    
    def get_trailing_history(self, hours_back: int = 24) -> List[Dict]:
        """Получение истории трейлинга"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            start_time = datetime.now() - timedelta(hours=hours_back)
            
            query = """
            SELECT * FROM trailing_logs 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
            """
            
            cursor = conn.cursor()
            cursor.execute(query, [start_time])
            rows = cursor.fetchall()
            
            # Получаем имена колонок
            columns = [description[0] for description in cursor.description]
            result = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return result
            
        except Exception as e:
            print(f"Error getting trailing history: {e}")
            return []