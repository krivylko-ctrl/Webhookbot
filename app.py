import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from bybit_api import BybitFuturesClient

# Настройка логирования
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("webhook_log.txt", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Инициализация Bybit клиента (DEMO режим)
bybit_client = BybitFuturesClient(testnet=True)

# Глобальные переменные для статистики
trade_stats = {
    "total_signals": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "last_signal_time": None,
    "last_signal_type": None,
    "active_positions": 0,
    "exchange": "Bybit DEMO"
}

recent_logs = []
MAX_LOGS = 100

# Флаг экстренной остановки
emergency_stop = False

# Защита от дублирования сигналов
last_signals = {}
SIGNAL_COOLDOWN = 5  # секунд между одинаковыми сигналами

def add_log(level, message):
    """Добавить запись в логи"""
    global recent_logs
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "level": level,
        "message": message
    }
    recent_logs.insert(0, log_entry)
    if len(recent_logs) > MAX_LOGS:
        recent_logs = recent_logs[:MAX_LOGS]

def check_signal_duplicate(signal_data):
    """Проверка на дублирование сигналов"""
    global last_signals
    
    signal_key = f"{signal_data.get('type')}_{signal_data.get('symbol')}_{signal_data.get('direction')}"
    current_time = datetime.now()
    
    if signal_key in last_signals:
        time_diff = (current_time - last_signals[signal_key]).total_seconds()
        if time_diff < SIGNAL_COOLDOWN:
            return False, f"Дублирующий сигнал заблокирован (прошло {time_diff:.1f}с, нужно {SIGNAL_COOLDOWN}с)"
    
    last_signals[signal_key] = current_time
    return True, "OK"

@app.before_request
def force_json_content_type():
    """Принудительно устанавливаем JSON content-type для webhook'ов"""
    if request.endpoint == 'webhook' and request.method == 'POST':
        if not request.is_json and request.data:
            try:
                # Пытаемся парсить как JSON даже если Content-Type неправильный
                json.loads(request.data.decode('utf-8'))
                request.environ['CONTENT_TYPE'] = 'application/json'
            except:
                pass

@app.route('/')
def index():
    """Главная страница с дашбордом"""
    return render_template('index.html', stats=trade_stats, recent_logs=recent_logs[:10])

@app.route('/logs')
def logs():
    """Страница логов"""
    return render_template('logs.html', logs=recent_logs)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка webhook сигналов от TradingView"""
    global trade_stats, emergency_stop
    
    if emergency_stop:
        add_log("WARNING", "🚨 ЭКСТРЕННАЯ ОСТАНОВКА АКТИВНА - сигнал проигнорирован")
        return jsonify({"status": "blocked", "reason": "emergency_stop"}), 423
    
    try:
        # Получение данных
        if request.is_json:
            data = request.get_json()
        else:
            # Попытка парсить JSON из raw data
            data = json.loads(request.data.decode('utf-8'))
        
        logger.info(f"📥 Получен webhook: {data}")
        add_log("INFO", f"📥 Webhook получен: {json.dumps(data, ensure_ascii=False)}")
        
        # Проверка на дублирование
        is_unique, duplicate_msg = check_signal_duplicate(data)
        if not is_unique:
            add_log("WARNING", f"🔄 {duplicate_msg}")
            return jsonify({"status": "duplicate", "message": duplicate_msg}), 409
        
        # Обновление статистики
        trade_stats["total_signals"] += 1
        trade_stats["last_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_stats["last_signal_type"] = data.get('type', 'unknown')
        
        # Обработка разных типов сигналов
        signal_type = data.get('type', '').lower()
        
        if signal_type == 'entry':
            result = handle_entry_signal(data)
        elif signal_type == 'trail_update':
            result = handle_trail_update(data)
        elif signal_type == 'exit':
            result = handle_exit_signal(data)
        elif signal_type == 'manual_close':
            result = handle_manual_close(data)
        else:
            result = {"error": f"Неизвестный тип сигнала: {signal_type}"}
            add_log("ERROR", f"❌ Неизвестный тип сигнала: {signal_type}")
        
        # Обновление статистики успеха/неудачи
        if "error" in result:
            trade_stats["failed_trades"] += 1
            add_log("ERROR", f"❌ Ошибка обработки сигнала: {result['error']}")
        else:
            trade_stats["successful_trades"] += 1
            add_log("SUCCESS", f"✅ Сигнал обработан успешно")
        
        return jsonify(result)
        
    except json.JSONDecodeError as e:
        error_msg = f"Ошибка парсинга JSON: {e}"
        logger.error(error_msg)
        add_log("ERROR", f"❌ {error_msg}")
        trade_stats["failed_trades"] += 1
        return jsonify({"error": error_msg}), 400
        
    except Exception as e:
        error_msg = f"Неожиданная ошибка: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        add_log("CRITICAL", f"💥 {error_msg}")
        trade_stats["failed_trades"] += 1
        return jsonify({"error": error_msg}), 500

def handle_entry_signal(data):
    """Обработка сигнала входа в позицию"""
    try:
        symbol = data.get('symbol', 'ETHUSDT')
        direction = data.get('direction', '').lower()
        quantity = float(data.get('qty', 0))
        entry_price = data.get('entry_price')
        stop_loss = data.get('stop_loss')
        take_profit = data.get('take_profit')
        
        if entry_price:
            entry_price = float(entry_price)
        if stop_loss:
            stop_loss = float(stop_loss)
        if take_profit:
            take_profit = float(take_profit)
        
        add_log("INFO", f"🎯 Вход в позицию: {direction.upper()} {quantity} {symbol}")
        
        # Размещение ордера через Bybit API
        result = bybit_client.place_order(
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        if result.get("retCode") == 0:
            trade_stats["active_positions"] += 1
            add_log("SUCCESS", f"✅ Позиция открыта: {direction.upper()} {quantity} {symbol}")
            return {"status": "success", "message": "Позиция открыта", "data": result}
        else:
            return {"error": f"Ошибка открытия позиции: {result.get('retMsg', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"Ошибка обработки сигнала входа: {str(e)}"}

def handle_trail_update(data):
    """Обработка обновления трейлинг стопа"""
    try:
        symbol = data.get('symbol', 'ETHUSDT')
        direction = data.get('direction', '').lower()
        new_trail_stop = float(data.get('new_trail_stop', 0))
        
        add_log("INFO", f"📈 Обновление трейлинг стопа: {symbol} -> {new_trail_stop}")
        
        result = bybit_client.update_trailing_stop(symbol, direction, new_trail_stop)
        
        if result.get("retCode") == 0:
            add_log("SUCCESS", f"✅ Трейлинг стоп обновлен: {new_trail_stop}")
            return {"status": "success", "message": "Трейлинг стоп обновлен"}
        else:
            return {"error": f"Ошибка обновления трейлинг стопа: {result.get('retMsg', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"Ошибка обновления трейлинг стопа: {str(e)}"}

def handle_exit_signal(data):
    """Обработка сигнала выхода из позиции"""
    try:
        symbol = data.get('symbol', 'ETHUSDT')
        direction = data.get('direction', '').lower()
        reason = data.get('reason', 'unknown')
        
        add_log("INFO", f"🚪 Выход из позиции: {symbol} ({reason})")
        
        result = bybit_client.close_position(symbol, direction)
        
        if result.get("retCode") == 0:
            if trade_stats["active_positions"] > 0:
                trade_stats["active_positions"] -= 1
            add_log("SUCCESS", f"✅ Позиция закрыта: {symbol} ({reason})")
            return {"status": "success", "message": f"Позиция закрыта ({reason})"}
        else:
            return {"error": f"Ошибка закрытия позиции: {result.get('retMsg', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"Ошибка закрытия позиции: {str(e)}"}

def handle_manual_close(data):
    """Обработка ручного закрытия позиции"""
    try:
        symbol = data.get('symbol', 'ETHUSDT')
        direction = data.get('direction', '').lower()
        
        add_log("INFO", f"✋ Ручное закрытие позиции: {symbol}")
        
        # Отменяем все ордера и закрываем позицию
        bybit_client.cancel_open_orders(symbol)
        result = bybit_client.close_position(symbol, direction)
        
        if result.get("retCode") == 0:
            if trade_stats["active_positions"] > 0:
                trade_stats["active_positions"] -= 1
            add_log("SUCCESS", f"✅ Позиция закрыта вручную: {symbol}")
            return {"status": "success", "message": "Позиция закрыта вручную"}
        else:
            return {"error": f"Ошибка ручного закрытия: {result.get('retMsg', 'Unknown error')}"}
            
    except Exception as e:
        return {"error": f"Ошибка ручного закрытия: {str(e)}"}

@app.route('/api/test', methods=['GET'])
def test_api():
    """Тестирование подключения к Bybit API"""
    try:
        ping_result = bybit_client.ping()
        account_info = bybit_client.get_account_info()
        positions = bybit_client.get_positions()
        
        add_log("INFO", "🔧 API тест выполнен")
        
        return jsonify({
            "ping": ping_result,
            "account": account_info,
            "positions": positions,
            "exchange": "Bybit DEMO"
        })
    except Exception as e:
        add_log("ERROR", f"❌ Ошибка API теста: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def toggle_emergency_stop():
    """Переключение экстренной остановки"""
    global emergency_stop
    emergency_stop = not emergency_stop
    
    status = "АКТИВИРОВАНА" if emergency_stop else "ДЕАКТИВИРОВАНА"
    message = f"🚨 Экстренная остановка {status}"
    
    add_log("WARNING" if emergency_stop else "INFO", message)
    flash(message, "warning" if emergency_stop else "success")
    
    return redirect(url_for('index'))

@app.route('/api/stats')
def get_stats():
    """API для получения статистики"""
    return jsonify(trade_stats)

@app.route('/api/refresh_data')
def refresh_data():
    """Обновление данных с биржи"""
    try:
        positions = bybit_client.get_positions()
        
        # Подсчет активных позиций
        active_count = 0
        if positions.get("retCode") == 0:
            position_list = positions.get("result", {}).get("list", [])
            active_count = len([p for p in position_list if float(p.get("size", 0)) > 0])
        
        trade_stats["active_positions"] = active_count
        add_log("INFO", f"📊 Данные обновлены: {active_count} активных позиций")
        
        flash("Данные успешно обновлены", "success")
        return redirect(url_for('index'))
        
    except Exception as e:
        error_msg = f"Ошибка обновления данных: {str(e)}"
        add_log("ERROR", f"❌ {error_msg}")
        flash(error_msg, "error")
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)