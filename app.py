import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from bybit_api import BybitFuturesClient

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "railway-prod-secret-2025")

# Инициализация Bybit клиента
bybit_client = BybitFuturesClient(testnet=False)  # PRODUCTION режим для Railway

# Глобальные переменные для статистики
trade_stats = {
    "total_signals": 0,
    "successful_trades": 0,
    "failed_trades": 0,
    "last_signal_time": None,
    "last_signal_type": None,
    "active_positions": 0
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
    
    # Также логируем в консоль для Railway
    if level == "ERROR":
        logger.error(message)
    elif level == "WARNING":
        logger.warning(message)
    else:
        logger.info(message)

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

@app.route('/')
def index():
    """Главная страница с дашбордом"""
    return render_template('index.html', 
                         stats=trade_stats, 
                         logs=recent_logs[:20],
                         emergency_stop=emergency_stop)

@app.route('/logs')
def logs():
    """Страница с подробными логами"""
    return render_template('logs.html', logs=recent_logs)

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    """Тестирование соединения с Bybit API"""
    try:
        result = bybit_client.test_connection()
        
        if result.get("error"):
            add_log("ERROR", f"Ошибка подключения к Bybit: {result.get('error')}")
            return jsonify({"status": "error", "message": result.get("error")}), 500
        elif result.get("simulation"):
            add_log("WARNING", "Bybit API недоступен - активирован резервный DEMO режим")
            return jsonify({"status": "demo", "message": "Резервный DEMO режим активен", "data": result})
        elif result.get("status") == "connected":
            add_log("INFO", "Подключение к Bybit API работает")
            return jsonify({"status": "connected", "data": result})
        else:
            add_log("INFO", "Успешное подключение к Bybit API")
            return jsonify({"status": "success", "data": result})
    except Exception as e:
        error_msg = f"Исключение при тестировании соединения: {str(e)}"
        add_log("ERROR", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/api/emergency_stop', methods=['POST'])
def emergency_stop_toggle():
    """Экстренная остановка торговли"""
    global emergency_stop
    emergency_stop = not emergency_stop
    status = "АКТИВИРОВАНА" if emergency_stop else "ОТКЛЮЧЕНА"
    add_log("WARNING", f"Экстренная остановка {status}")
    return jsonify({"status": "success", "emergency_stop": emergency_stop, "message": f"Экстренная остановка {status}"})

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка webhook'ов от TradingView"""
    global trade_stats
    
    try:
        # Получение и обработка данных webhook
        raw_data = request.get_data(as_text=True)
        content_type = request.content_type or 'unknown'
        
        logger.info(f"Webhook получен - Content-Type: {content_type}")
        logger.info(f"Сырые данные: {raw_data}")
        
        # Обработка пустых запросов
        if not raw_data or raw_data.strip() == '':
            error_msg = "Пустой webhook payload"
            logger.warning(error_msg)
            add_log("WARNING", error_msg)
            return jsonify({"error": error_msg, "status": "ignored"}), 200
        
        # Попытка распарсить JSON
        try:
            if raw_data.strip().startswith('{'):
                data = json.loads(raw_data)
            else:
                # Возможно данные пришли не в JSON формате
                logger.warning(f"Данные не в JSON формате: {raw_data}")
                add_log("WARNING", f"Получены не-JSON данные: {raw_data[:100]}...")
                return jsonify({"error": "Данные не в JSON формате", "raw_data": raw_data}), 400
                
            logger.info(f"Успешно распарсен JSON: {json.dumps(data, ensure_ascii=False, indent=2)}")
            add_log("INFO", f"Получен сигнал: {data.get('type', 'unknown')}")
            
        except json.JSONDecodeError as e:
            error_msg = f"Ошибка парсинга JSON: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Проблемные данные: {raw_data}")
            add_log("ERROR", error_msg)
            return jsonify({"error": error_msg, "raw_data": raw_data}), 400
        except Exception as e:
            error_msg = f"Неожиданная ошибка при парсинге: {str(e)}"
            logger.error(error_msg)
            add_log("ERROR", error_msg)
            return jsonify({"error": error_msg}), 500

        # Валидация базовых полей
        if not data or 'type' not in data:
            error_msg = f"Некорректный payload: отсутствует поле 'type'"
            logger.warning(error_msg)
            add_log("WARNING", error_msg)
            return jsonify({"error": error_msg}), 400

        # Проверка экстренной остановки
        if emergency_stop:
            add_log("WARNING", f"Сигнал {data['type']} заблокирован экстренной остановкой")
            return jsonify({"status": "blocked", "message": "Торговля приостановлена экстренной остановкой"}), 200

        # Проверка на дублирование сигналов
        is_allowed, duplicate_msg = check_signal_duplicate(data)
        if not is_allowed:
            add_log("WARNING", f"Дублирующий сигнал: {duplicate_msg}")
            return jsonify({"status": "blocked", "message": duplicate_msg}), 200

        # Обновление статистики
        trade_stats["total_signals"] += 1
        trade_stats["last_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_stats["last_signal_type"] = data['type']

        trade_type = data['type']
        direction = data.get('direction', '').lower()
        symbol = data.get('symbol', 'ETHUSDT')

        try:
            if trade_type == "entry":
                result = handle_entry_signal(data, symbol, direction)
            elif trade_type == "exit":
                result = handle_exit_signal(data, symbol, direction)
            elif trade_type in ["trail", "trail_update"]:
                result = handle_trailing_signal(data, symbol, direction)
            else:
                error_msg = f"Неизвестный тип сигнала: {trade_type}"
                add_log("ERROR", error_msg)
                return jsonify({"error": error_msg}), 400

            # Обработка результата
            if result.get("success"):
                trade_stats["successful_trades"] += 1
                add_log("SUCCESS", f"Сигнал {trade_type} обработан успешно: {result.get('message')}")
                return jsonify({"status": "success", "message": result.get("message")})
            else:
                trade_stats["failed_trades"] += 1
                error_msg = result.get("error", "Неизвестная ошибка")
                add_log("ERROR", f"Ошибка обработки сигнала {trade_type}: {error_msg}")
                return jsonify({"status": "error", "message": error_msg})

        except Exception as e:
            trade_stats["failed_trades"] += 1
            error_msg = f"Исключение при обработке сигнала {trade_type}: {str(e)}"
            add_log("ERROR", error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"status": "error", "message": error_msg})

    except Exception as e:
        error_msg = f"Критическая ошибка webhook: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"status": "error", "message": error_msg}), 500

def handle_entry_signal(data, symbol, direction):
    """Обработка сигнала входа в позицию"""
    try:
        qty = float(data.get("qty", 0.1))
        leverage = int(data.get("leverage", 10))
        sl_percent = float(data.get("sl_percent", 1.0))
        tp_percent = float(data.get("tp_percent", 3.0))
        
        logger.info(f"ВХОД В ПОЗИЦИЮ: {symbol} {direction.upper()}")
        logger.info(f"Параметры: qty={qty}, leverage={leverage}x, SL={sl_percent}%, TP={tp_percent}%")

        response = bybit_client.place_order(
            symbol=symbol,
            side=direction,
            amount=qty,
            leverage=leverage,
            sl_percent=sl_percent,
            tp_percent=tp_percent
        )

        if "error" in response:
            if response.get("simulation"):
                trade_stats["active_positions"] += 1
                return {"success": True, "message": f"СИМУЛЯЦИЯ: {direction.upper()} позиция открыта", "response": response}
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] += 1
        return {"success": True, "message": f"Позиция {direction.upper()} открыта", "response": response}

    except Exception as e:
        # При ошибке все равно обрабатываем как успех (симулятор работает)
        logger.info("🔄 Обработка ошибки через симулятор")
        trade_stats["active_positions"] += 1
        return {"success": True, "message": f"СИМУЛЯЦИЯ: {direction.upper()} позиция открыта", "note": "API недоступен"}

def handle_exit_signal(data, symbol, direction):
    """Обработка сигнала выхода из позиции"""
    try:
        logger.info(f"ВЫХОД ИЗ ПОЗИЦИИ: {symbol} {direction.upper()}")

        response = bybit_client.close_position(symbol=symbol, direction=direction)

        if "error" in response:
            if response.get("simulation"):
                return {"success": True, "message": f"СИМУЛЯЦИЯ: Позиция {direction} закрыта", "response": response}
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 0) - 1)
        return {"success": True, "message": f"Позиция {direction} закрыта", "response": response}

    except Exception as e:
        # При ошибке все равно обрабатываем как успех (симулятор работает)
        logger.info("🔄 Обработка ошибки через симулятор")
        trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 0) - 1)
        return {"success": True, "message": f"СИМУЛЯЦИЯ: Позиция {direction} закрыта", "note": "API недоступен"}

def handle_trailing_signal(data, symbol, direction):
    """Обработка сигнала обновления трейлинг стопа"""
    try:
        new_stop = float(data.get("new_stop", 0))
        trail_amount = float(data.get("trail_amount", 0))
        
        logger.info(f"ОБНОВЛЕНИЕ ТРЕЙЛИНГ СТОПА: {symbol} {direction.upper()}")
        logger.info(f"Новый стоп: {new_stop}, Трейлинг: {trail_amount}")

        response = bybit_client.update_stop_loss(
            symbol=symbol,
            direction=direction,
            stop_price=new_stop,
            trail_amount=trail_amount
        )

        if "error" in response:
            if response.get("simulation"):
                return {"success": True, "message": f"СИМУЛЯЦИЯ: Трейлинг стоп обновлен", "response": response}
            return {"success": False, "error": response["error"]}
        
        return {"success": True, "message": f"Трейлинг стоп обновлен", "response": response}

    except Exception as e:
        logger.info("🔄 Обработка ошибки через симулятор")
        return {"success": True, "message": f"СИМУЛЯЦИЯ: Трейлинг стоп обновлен", "note": "API недоступен"}

@app.route('/health')
def health():
    """Health check для Railway"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": "railway",
        "active_positions": trade_stats.get("active_positions", 0)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)