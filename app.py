import os
import json
import logging
import traceback
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from mexc_api import MexcFuturesClient

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

# Инициализация MEXC клиента
mexc_client = MexcFuturesClient()

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
    """Тестирование соединения с MEXC API"""
    try:
        result = mexc_client.get_account_info()
        if "error" in result:
            add_log("ERROR", f"Ошибка подключения к MEXC: {result['error']}")
            return jsonify({"status": "error", "message": result["error"]}), 500
        else:
            add_log("INFO", "Успешное подключение к MEXC API")
            return jsonify({"status": "success", "data": result})
    except Exception as e:
        error_msg = f"Исключение при тестировании соединения: {str(e)}"
        add_log("ERROR", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/webhook', methods=['POST'])
def webhook():
    """Обработка webhook'ов от TradingView"""
    global trade_stats
    
    try:
        # Попытка получить JSON данные
        try:
            data = request.get_json(force=True)
            logger.info(f"Получен webhook: {json.dumps(data, ensure_ascii=False, indent=2)}")
            add_log("INFO", f"Получен сигнал: {data.get('type', 'unknown')}")
        except Exception as e:
            # Если не удалось распарсить JSON, попробуем как текст
            raw_data = request.get_data(as_text=True)
            logger.error(f"Ошибка парсинга JSON: {e}")
            logger.error(f"Сырые данные: {raw_data}")
            add_log("ERROR", f"Ошибка парсинга JSON: {str(e)}")
            return jsonify({"error": "Неверный формат JSON", "raw_data": raw_data}), 400

        # Валидация базовых полей
        if not data or 'type' not in data:
            error_msg = f"Некорректный payload: отсутствует поле 'type'"
            logger.warning(error_msg)
            add_log("WARNING", error_msg)
            return jsonify({"error": error_msg}), 400

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
    """Обработка сигнала входа в позицию"""
    try:
        qty = float(data.get("qty", 0))
        entry_price = float(data.get("entry_price", 0))
        stop_loss = float(data.get("stop_loss", 0))
        take_profit = float(data.get("take_profit", 0))
        leverage = 30  # Фиксированное плечо

        if qty <= 0 or entry_price <= 0:
            return {"success": False, "error": "Некорректные параметры входа"}

        logger.info(f"ВХОД В ПОЗИЦИЮ: {symbol} {direction.upper()}")
        logger.info(f"Количество: {qty}, Цена: {entry_price}, SL: {stop_loss}, TP: {take_profit}")

        response = mexc_client.open_position(
            symbol=symbol,
            direction=direction,
            quantity=qty,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage
        )

        if "error" in response:
            # Особая обработка для демо режима
            if response.get("demo_mode"):
                return {"success": True, "message": f"ДЕМО: Позиция {direction} открыта (API недоступен)", "response": response}
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = trade_stats.get("active_positions", 0) + 1
        return {"success": True, "message": f"Позиция {direction} открыта", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка входа: {str(e)}"}

def handle_exit_signal(data, symbol, direction):
    """Обработка сигнала выхода из позиции"""
    try:
        logger.info(f"ВЫХОД ИЗ ПОЗИЦИИ: {symbol} {direction.upper()}")

        response = mexc_client.close_position(symbol=symbol, direction=direction)

        if "error" in response:
            if response.get("demo_mode"):
                return {"success": True, "message": f"ДЕМО: Позиция {direction} закрыта (API недоступен)", "response": response}
            return {"success": False, "error": response["error"]}
        
        trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 0) - 1)
        return {"success": True, "message": f"Позиция {direction} закрыта", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка выхода: {str(e)}"}

def handle_trailing_signal(data, symbol, direction):
    """Обработка сигнала обновления трейлинг стопа"""
    try:
        new_stop = float(data.get("stop_loss") or data.get("new_trail_stop") or 0)
        
        if new_stop <= 0:
            return {"success": False, "error": "Некорректная цена стопа"}

        logger.info(f"ОБНОВЛЕНИЕ ТРЕЙЛИНГ СТОПА: {symbol} {direction.upper()}, новый стоп: {new_stop}")

        response = mexc_client.edit_position(
            symbol=symbol,
            direction=direction,
            stop_loss_price=new_stop
        )

        if "error" in response:
            if response.get("demo_mode"):
                return {"success": True, "message": f"ДЕМО: Трейлинг стоп обновлен (API недоступен)", "response": response}
            return {"success": False, "error": response["error"]}
        
        return {"success": True, "message": f"Трейлинг стоп обновлен", "response": response}

    except Exception as e:
        return {"success": False, "error": f"Ошибка обновления стопа: {str(e)}"}

def handle_manual_close(data, symbol, direction):
    """Обработка ручного закрытия позиции"""
    try:
        logger.info(f"РУЧНОЕ ЗАКРЫТИЕ: {symbol} {direction.upper()}")

        response = mexc_client.close_position(symbol=symbol, direction=direction)

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
