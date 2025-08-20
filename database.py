import sqlite3
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import pandas as pd

class Database:
    """Управление базой данных SQLite для торгового бота"""
    
    def __init__(self, db_path: str = "kwin_bot.db", memory: bool = False):
        if memory:
            self.db_path = ":memory:"
        else:
            self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Таблица сделок
            cursor.execute('''
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
                    entry_time TIMESTAMP NOT NULL,
                    exit_time TIMESTAMP,
                    exit_reason TEXT,
                    status TEXT NOT NULL DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица equity
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS equity_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    equity REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица состояния бота
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS bot_state (
                    id INTEGER PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица конфигурации
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS config (
                    id INTEGER PRIMARY KEY,
                    config_data TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Таблица логов
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL,
                    module TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def save_trade(self, trade_data: Dict) -> int:
        """Сохранение новой сделки"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO trades (
                    symbol, direction, entry_price, stop_loss, take_profit,
                    quantity, entry_time, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data.get('symbol'),
                trade_data.get('direction'),
                trade_data.get('entry_price'),
                trade_data.get('stop_loss'),
                trade_data.get('take_profit'),
                trade_data.get('quantity'),
                trade_data.get('entry_time'),
                trade_data.get('status', 'open')
            ))
            
            trade_id = cursor.lastrowid
            conn.commit()
            return trade_id if trade_id is not None else 0
    
    def update_trade_exit(self, trade_data: Dict, fee_rate: float = 0.00055):
        """Обновление данных выхода из сделки с автоматическим расчетом PnL/RR"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Находим открытую сделку
            trade_id = trade_data.get('trade_id')
            if trade_id:
                cursor.execute('SELECT * FROM trades WHERE id = ? AND status = "open"', (trade_id,))
            else:
                cursor.execute('SELECT * FROM trades WHERE status = "open" ORDER BY entry_time DESC LIMIT 1')
            
            trade = cursor.fetchone()
            if not trade:
                print("No open trade found to update")
                return
            
            # Получаем данные сделки
            columns = [desc[0] for desc in cursor.description]
            trade_dict = dict(zip(columns, trade))
            
            entry_price = float(trade_dict['entry_price'])
            stop_loss = float(trade_dict['stop_loss']) if trade_dict['stop_loss'] else None
            take_profit = float(trade_dict['take_profit']) if trade_dict['take_profit'] else None
            quantity = float(trade_dict['quantity'])
            direction = trade_dict['direction']
            exit_price = float(trade_data.get('exit_price', 0))
            
            # Расчет PnL с двойной комиссией (вход + выход)
            if direction == 'long':
                gross_pnl = (exit_price - entry_price) * quantity
            else:  # short
                gross_pnl = (entry_price - exit_price) * quantity
            
            # Комиссии
            entry_fee = entry_price * quantity * fee_rate
            exit_fee = exit_price * quantity * fee_rate
            total_fees = entry_fee + exit_fee
            
            # Чистый PnL
            net_pnl = gross_pnl - total_fees
            
            # Расчет Risk/Reward
            rr = 0.0
            if stop_loss:
                risk = abs(entry_price - stop_loss) * quantity
                if risk > 0:
                    rr = abs(gross_pnl) / risk
            
            # Обновляем сделку
            cursor.execute('''
                UPDATE trades 
                SET exit_price = ?, exit_time = ?, exit_reason = ?, 
                    pnl = ?, rr = ?, status = ?
                WHERE id = ?
            ''', (
                exit_price,
                trade_data.get('exit_time'),
                trade_data.get('exit_reason'),
                net_pnl,
                rr,
                trade_data.get('status', 'closed'),
                trade_dict['id']
            ))
            
            conn.commit()
            print(f"Trade updated: PnL={net_pnl:.2f}, RR={rr:.2f}, Fees={total_fees:.2f}")
    
    def get_recent_trades(self, limit: int = 50) -> List[Dict]:
        """Получение последних сделок"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM trades 
                ORDER BY entry_time DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def get_trades_by_period(self, days: int) -> List[Dict]:
        """Получение сделок за период"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT * FROM trades 
                WHERE entry_time >= ?
                ORDER BY entry_time DESC
            ''', (start_date,))
            
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def save_equity_snapshot(self, equity: float):
        """Сохранение снапшота equity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO equity_history (equity)
                VALUES (?)
            ''', (equity,))
            
            conn.commit()
    
    def get_equity_history(self, days: int = 30) -> List[Dict]:
        """Получение истории equity"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            cursor.execute('''
                SELECT equity, timestamp FROM equity_history 
                WHERE timestamp >= ?
                ORDER BY timestamp ASC
            ''', (start_date,))
            
            rows = cursor.fetchall()
            
            return [{'equity': row[0], 'timestamp': row[1]} for row in rows]
    
    def save_bot_state(self, state: Dict):
        """Сохранение состояния бота"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            state_json = json.dumps(state)
            
            cursor.execute('''
                INSERT OR REPLACE INTO bot_state (id, state_data)
                VALUES (1, ?)
            ''', (state_json,))
            
            conn.commit()
    
    def get_bot_state(self) -> Optional[Dict]:
        """Получение состояния бота"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('SELECT state_data FROM bot_state WHERE id = 1')
            row = cursor.fetchone()
            
            if row:
                return json.loads(row[0])
            return None
    
    def get_performance_stats(self, days: int = 30) -> Dict:
        """Получение статистики производительности"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            start_date = datetime.now() - timedelta(days=days)
            
            # Общая статистика
            cursor.execute('''
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
            ''', (start_date,))
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:  # Есть сделки
                total_trades = row[0]
                winning_trades = row[1] or 0
                
                stats = {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'losing_trades': total_trades - winning_trades,
                    'win_rate': (winning_trades / total_trades) * 100 if total_trades > 0 else 0,
                    'avg_pnl': row[2] or 0,
                    'total_pnl': row[3] or 0,
                    'avg_rr': row[4] or 0,
                    'max_win': row[5] or 0,
                    'max_loss': row[6] or 0,
                    'avg_hold_time': row[7] or 0
                }
            else:
                # Нет сделок
                stats = {
                    'total_trades': 0,
                    'winning_trades': 0,
                    'losing_trades': 0,
                    'win_rate': 0,
                    'avg_pnl': 0,
                    'total_pnl': 0,
                    'avg_rr': 0,
                    'max_win': 0,
                    'max_loss': 0,
                    'avg_hold_time': 0
                }
            
            return stats
    
    def get_trades_count_today(self) -> int:
        """Получение количества сделок за сегодня"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute('''
                SELECT COUNT(*) FROM trades 
                WHERE DATE(entry_time) = ?
            ''', (today,))
            
            return cursor.fetchone()[0]
    
    def get_pnl_today(self) -> float:
        """Получение PnL за сегодня"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            today = datetime.now().date()
            
            cursor.execute('''
                SELECT COALESCE(SUM(pnl), 0) FROM trades 
                WHERE DATE(entry_time) = ? AND status = 'closed'
            ''', (today,))
            
            return cursor.fetchone()[0]
    
    def save_log(self, level: str, message: str, module: str = ""):
        """Сохранение лога"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO logs (level, message, module)
                VALUES (?, ?, ?)
            ''', (level, message, module or ""))
            
            conn.commit()
    
    def get_logs(self, limit: int = 100) -> List[Dict]:
        """Получение логов"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT level, message, module, timestamp FROM logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            columns = ['level', 'message', 'module', 'timestamp']
            rows = cursor.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
    
    def cleanup_old_data(self, days_to_keep: int = 90):
        """Очистка старых данных"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            # Удаляем старые логи
            cursor.execute('DELETE FROM logs WHERE timestamp < ?', (cutoff_date,))
            
            # Удаляем старую историю equity (оставляем только closed сделки)
            cursor.execute('DELETE FROM equity_history WHERE timestamp < ?', (cutoff_date,))
            
            conn.commit()
    
    def add_trade(self, trade_data: Dict) -> int:
        """Добавить сделку в базу и вернуть ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO trades 
                    (symbol, direction, entry_price, exit_price, stop_loss, take_profit, quantity, 
                     pnl, rr, entry_time, exit_time, exit_reason, status) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trade_data['symbol'],
                    trade_data['direction'],
                    trade_data['entry_price'],
                    trade_data.get('exit_price'),
                    trade_data.get('stop_loss'),
                    trade_data.get('take_profit'),
                    trade_data['quantity'],
                    trade_data.get('pnl', 0),
                    trade_data.get('rr', 0),
                    trade_data['entry_time'],
                    trade_data.get('exit_time'),
                    trade_data.get('exit_reason'),
                    trade_data['status']
                ))
                
                trade_id = cursor.lastrowid
                conn.commit()
                return int(trade_id or 0)
                
        except Exception as e:
            print(f"Error adding trade: {e}")
            return 0
            
    def update_trade_stop_loss(self, trade_id: int, new_stop_loss: float):
        """Обновить stop_loss для сделки"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE trades SET stop_loss = ? WHERE id = ?
                """, (new_stop_loss, trade_id))
                conn.commit()
        except Exception as e:
            print(f"Error updating stop loss: {e}")
    
    def get_all_trades(self) -> List[Dict]:
        """Получить все сделки"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM trades ORDER BY entry_time DESC")
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            print(f"Error getting all trades: {e}")
            return []
