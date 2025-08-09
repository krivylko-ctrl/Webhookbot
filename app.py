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

# Инициализация Bybit клиента (ТОЛЬКО TESTNET)
bybit_client = BybitFuturesClient(testnet=True)

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
    logger.log(getattr(logging, level), message)

def check_signal_duplicate(data):
    """Проверка на дублирующиеся сигналы"""
    signal_key = f"{data.get('type')}_{data.get('symbol')}_{data.get('direction')}"
    current_time = datetime.now()
    
    if signal_key in last_signals:
        time_diff = (current_time - last_signals[signal_key]).seconds
        if time_diff < SIGNAL_COOLDOWN:
            return False, f"Дублирующий сигнал {signal_key} (прошло {time_diff}с)"
    
    last_signals[signal_key] = current_time
    return True, "OK"

@app.before_request
def force_json_content_type():
    """Принудительно устанавливаем JSON Content-Type для webhook'ов"""
    if request.path == '/webhook' and request.method == 'POST':
        if request.content_type == 'text/plain' or not request.content_type:
            request.environ['CONTENT_TYPE'] = 'application/json'

@app.route('/')
def index():
    """Главная страница с мониторингом бота"""
    return render_template('index.html', stats=trade_stats, recent_logs=recent_logs[:10])

@app.route('/logs')
def logs():
    """Страница с детальными логами"""
    return render_template('logs.html', logs=recent_logs)

@app.route('/api/stats')
def api_stats():
    """API для получения статистики"""
    return jsonify(trade_stats)

@app.route('/api/test_connection')
def test_connection():
    """Тестирование соединения с Bybit testnet API"""
    try:
        result = bybit_client.test_connection()
        if "error" in result:
            add_log("ERROR", f"Ошибка подключения к Bybit testnet: {result['error']}")
            return jsonify({"status": "error", "message": result["error"]}), 500
        elif result.get("status") == "connected":
            add_log("SUCCESS", "Подключение к Bybit testnet API работает")
            return jsonify({"status": "connected", "data": result})
        else:
            add_log("ERROR", f"Неизвестный статус подключения: {result}")
            return jsonify({"status": "unknown", "data": result}), 500
    except Exception as e:
        error_msg = f"Исключение при тестировании соединения: {str(e)}"
        add_log("ERROR", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/health')
def health():
    """Проверка состояния системы"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_positions": trade_stats.get("active_positions", 0),
        "environment": "railway"
    })

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
            elif trade_type == "manual_close":
                result = handle_manual_close(data, symbol, direction)
            else:
                error_msg = f"Неизвестный тип сигнала: {trade_type}"
                logger.warning(error_msg)
                add_log("WARNING", error_msg)
                return jsonify({"error": error_msg}), 400

            if result.get("success"):
                trade_stats["successful_trades"] += 1
                add_log("SUCCESS", f"Успешно обработан сигнал {trade_type}")
                return jsonify({"status": "success", "message": result.get("message")}), 200
            else:
                trade_stats["failed_trades"] += 1
                add_log("ERROR", f"Ошибка обработки сигнала {trade_type}: {result.get('error')}")
                return jsonify({"status": "error", "message": result.get("error")}), 500

        except Exception as e:
            trade_stats["failed_trades"] += 1
            error_msg = f"Исключение при обработке {trade_type}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            add_log("ERROR", error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = f"Критическая ошибка webhook: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        add_log("CRITICAL", error_msg)
        return jsonify({"error": error_msg}), 500

def handle_entry_signal(data, symbol, direction):
    """Обработка сигнала входа в позицию - ТОЛЬКО РЕАЛЬНЫЙ API"""
    try:
        qty = float(data.get("qty", 0))
        entry_price = float(data.get("entry_price", 0))
        stop_loss = float(data.get("stop_loss", 0))
        take_profit = float(data.get("take_profit", 0))
        leverage = 30  # Фиксированное плечо

        # Проверки безопасности
        if qty <= 0 or entry_price <= 0:
            return {"success": False, "error": "Некорректные параметры входа"}
        
        # Проверка разумности стоп-лосса
        if direction.lower() == "long" and stop_loss >= entry_price:
            return {"success": False, "error": "Некорректный стоп-лосс для лонга"}
        elif direction.lower() == "short" and stop_loss <= entry_price:
            return {"success": False, "error": "Некорректный стоп-лосс для шорта"}

        logger.info(f"ВХОД В ПОЗИЦИЮ: {symbol} {direction.upper()}")
        logger.info(f"Количество: {qty}, Цена: {entry_price}, SL: {stop_loss}, TP: {take_profit}")

        response = bybit_client.open_position(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage
        )

        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = trade_stats.get("active_positions", 0) + 1
        return {"success": True, "message": f"Позиция {direction} открыта", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка входа: {str(e)}"}

def handle_exit_signal(data, symbol, direction):
    """Обработка сигнала выхода из позиции - ТОЛЬКО РЕАЛЬНЫЙ API"""
    try:
        logger.info(f"ВЫХОД ИЗ ПОЗИЦИИ: {symbol} {direction.upper()}")

        response = bybit_client.close_position(symbol=symbol, direction=direction)

        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 0) - 1)
        return {"success": True, "message": f"Позиция {direction} закрыта", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка выхода: {str(e)}"}

def handle_trailing_signal(data, symbol, direction):
    """Обработка сигнала обновления трейлинг стопа - ТОЛЬКО РЕАЛЬНЫЙ API"""
    try:
        new_stop = float(data.get("stop_loss") or data.get("new_trail_stop") or 0)
        
        if new_stop <= 0:
            return {"success": False, "error": "Некорректная цена стопа"}

        logger.info(f"ОБНОВЛЕНИЕ ТРЕЙЛИНГ СТОПА: {symbol} {direction.upper()}, новый стоп: {new_stop}")

        response = bybit_client.update_stop_loss(
            symbol=symbol,
            direction=direction,
            stop_price=new_stop
        )

        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        return {"success": True, "message": f"Трейлинг стоп обновлен", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка обновления стопа: {str(e)}"}

def handle_manual_close(data, symbol, direction):
    """Обработка ручного закрытия позиции - ТОЛЬКО РЕАЛЬНЫЙ API"""
    try:
        logger.info(f"РУЧНОЕ ЗАКРЫТИЕ: {symbol} {direction.upper()}")

        response = bybit_client.close_position(symbol=symbol, direction=direction)

        if "error" in response:
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 0) - 1)
        return {"success": True, "message": f"Позиция {direction} закрыта вручную", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка ручного закрытия: {str(e)}"}

@app.errorhandler(404)
def not_found(error):
    """Обработка 404 ошибок"""
    return render_template('index.html', stats=trade_stats, recent_logs=recent_logs[:10]), 404

@app.errorhandler(500)
def internal_error(error):
    """Обработка 500 ошибок"""
    logger.error(f"Внутренняя ошибка сервера: {error}")
    add_log("ERROR", f"Внутренняя ошибка сервера: {str(error)}")
    return render_template('index.html', stats=trade_stats, recent_logs=recent_logs[:10]), 500

if __name__ == '__main__':
    logger.info("Запуск торгового бота...")
    add_log("INFO", "Торговый бот запущен")
    app.run(host='0.0.0.0', port=5000, debug=True)