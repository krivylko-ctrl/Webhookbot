import os
import json
import logging
import traceback
import hashlib
import time
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, render_template, flash, redirect, url_for
from bybit_v5_fixed import execute_trade_signal, test_connection, normalize_symbol, mask
from trail_engine import TrailEngine

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
app.secret_key = os.environ.get("SESSION_SECRET", "railway-production-secret-2025")

# Инициализация trail engine для продвинутого трейлинга
trail_engine = TrailEngine(poll_sec=1.0)

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

# Защита от дубликатов по 15-мин барам
last_bar_entries = {}
SIGNAL_COOLDOWN = 5  # секунд между одинаковыми сигналами

# Фильтр точных дублей по отпечатку payload на 60 сек
RECENT_HASHES = {}
DEDUP_TTL_SEC = 60

def payload_fingerprint(d: dict) -> str:
    keys = ["type","symbol","direction","qty","entry_price","stop_loss","take_profit","leverage"]
    norm = {k: d.get(k) for k in keys}
    blob = json.dumps(norm, sort_keys=True, separators=(",",":"))
    return hashlib.sha256(blob.encode()).hexdigest()

def is_duplicate_exact(d: dict) -> bool:
    fp = payload_fingerprint(d)
    now = time.time()
    for h, ts in list(RECENT_HASHES.items()):
        if now - ts > DEDUP_TTL_SEC:
            RECENT_HASHES.pop(h, None)
    if fp in RECENT_HASHES:
        return True
    RECENT_HASHES[fp] = now
    return False

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

def _start_trail_engine():
    """Стартуем фоновый цикл трейла"""
    try:
        trail_engine.start()
        logger.info("🎯 Trail engine started successfully")
    except Exception as e:
        logger.exception(f"Trail engine start failed: {e}")

# Запускаем trail engine сразу при инициализации
_start_trail_engine()

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
def api_test_connection():
    """Тестирование соединения с Bybit V5 PRODUCTION API"""
    try:
        result = test_connection()
        if result.get("status") == "connected":
            add_log("SUCCESS", "RAILWAY: Подключение к Bybit V5 API работает")
            return jsonify(result)
        else:
            add_log("ERROR", f"RAILWAY: Ошибка подключения V5: {result.get('message')}")
            return jsonify(result), 500
    except Exception as e:
        error_msg = f"RAILWAY: Исключение при тестировании V5: {str(e)}"
        add_log("ERROR", error_msg)
        return jsonify({"status": "error", "message": error_msg}), 500

@app.route('/health')
def health():
    """Проверка состояния системы"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_positions": trade_stats.get("active_positions", 0),
        "environment": "production",
        "mode": "mainnet",
        "platform": "railway"
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
        
        logger.info(f"RAILWAY WEBHOOK получен - Content-Type: {content_type}")
        logger.info(f"Сырые данные: {raw_data}")
        
        # Обработка пустых запросов
        if not raw_data or raw_data.strip() == '':
            error_msg = "Пустой webhook payload"
            logger.warning(error_msg)
            add_log("WARNING", error_msg)
            return jsonify({"error": error_msg, "status": "ignored"}), 200
        
        try:
            data = json.loads(request.data.decode("utf-8"))
            # нормализация символа .P -> без .P
            if isinstance(data.get("symbol"), str) and data["symbol"].endswith(".P"):
                original_symbol = data["symbol"]
                data["symbol"] = data["symbol"][:-2]
                logger.info(f"RAILWAY нормализация символа: {original_symbol} -> {data['symbol']}")
            logger.info(f"RAILWAY успешно распарсен JSON: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # Фильтр точных дублей по отпечатку payload на 60 сек
            if is_duplicate_exact(data):
                msg = "Дубликат идентичного сигнала за 60с — игнор"
                return jsonify({"status":"blocked","message":msg}), 200
            
            add_log("INFO", f"Получен сигнал: {data.get('type', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"RAILWAY невалидный JSON: {e}")
            logger.error(f"Сырые данные: {raw_data[:200]}...")
            add_log("ERROR", f"Невалидный JSON: {str(e)}")
            return jsonify({"error": "Невалидный JSON", "raw_data": raw_data[:100]}), 400

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
        
        # Безопасный парсинг полей с дефолтами
        signal_type = (data.get("type") or "").lower()
        symbol = data.get("symbol", "ETHUSDT")
        side = data.get("side") or data.get("direction", "")
        
        # Безопасное извлечение числовых полей
        tp = float(data.get("take_profit")) if data.get("take_profit") is not None else None
        sl = float(data.get("stop_loss")) if data.get("stop_loss") is not None else None
        qty = float(data.get("qty")) if data.get("qty") is not None else None
        
        # Валидация минимальных полей для каждого типа
        if signal_type == "entry":
            for field in ["symbol", "direction", "qty"]:
                if not data.get(field):
                    error_msg = f"Missing field '{field}' for entry signal"
                    add_log("ERROR", error_msg)
                    return jsonify({"error": error_msg}), 400
        elif signal_type in ("trail_update", "exit"):
            for field in ["symbol", "direction"]:
                if not data.get(field):
                    error_msg = f"Missing field '{field}' for {signal_type} signal"
                    add_log("ERROR", error_msg)
                    return jsonify({"error": error_msg}), 400
        
        # Защита от дубликатов по 15-мин барам для entry
        if signal_type == "entry":
            bar_time = int(data.get("bar_time", 0))
            bar_key = (symbol, side)
            if last_bar_entries.get(bar_key) == bar_time and bar_time > 0:
                msg = f"Duplicate entry for same 15m bar: {symbol} {side}"
                add_log("WARNING", msg)
                return jsonify({"status": "blocked", "message": msg}), 200
            last_bar_entries[bar_key] = bar_time
        
        # Интеграция с trail engine для продвинутого трейлинга
        try:
            if signal_type == "entry":
                trail_engine.start_from_entry(data)
                logger.info(f"🎯 Trail engine: start tracking for {symbol} {side}")
            elif signal_type in ["trail_update", "trail"]:
                accepted = trail_engine.on_external_trail(data)
                status = "принят" if accepted else "отклонен (хуже текущего)"
                logger.info(f"🎯 Trail engine: external update {status}")
            elif signal_type == "exit":
                trail_engine.on_exit(data)
                logger.info(f"🎯 Trail engine: position closed for {symbol} {side}")
        except Exception as e:
            logger.warning(f"Trail engine error: {e}")  # не прерываем основной flow
        trade_stats["last_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_stats["last_signal_type"] = data['type']

        trade_type = data['type']
        direction = data.get('direction', '').lower()
        symbol = data.get('symbol', 'ETHUSDT')

        try:
            if trade_type == "entry":
                result = handle_entry_signal(data, symbol, direction)
            elif trade_type == "exit":
                # Используем execute_trade_signal из bybit_v5_fixed
                response = execute_trade_signal({
                    "type": "exit",
                    "symbol": symbol,
                    "direction": direction,
                    "reason": data.get("reason", "")
                })
                if response.get("status") == "success":
                    trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 1) - 1)
                    result = {"success": True, "message": response.get("message")}
                else:
                    result = {"success": False, "error": response.get("message")}
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
                add_log("SUCCESS", f"RAILWAY: Успешно обработан сигнал {trade_type}")
                return jsonify({"status": "success", "message": result.get("message"), "platform": "railway"}), 200
            else:
                trade_stats["failed_trades"] += 1
                add_log("ERROR", f"RAILWAY: Ошибка обработки сигнала {trade_type}: {result.get('error')}")
                return jsonify({"status": "error", "message": result.get("error"), "platform": "railway"}), 500

        except Exception as e:
            trade_stats["failed_trades"] += 1
            error_msg = f"RAILWAY: Исключение при обработке {trade_type}: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            add_log("ERROR", error_msg)
            return jsonify({"error": error_msg, "platform": "railway"}), 500

    except Exception as e:
        error_msg = f"RAILWAY: Критическая ошибка webhook: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        add_log("CRITICAL", error_msg)
        return jsonify({"error": error_msg, "platform": "railway"}), 500

def handle_entry_signal(data, symbol, direction):
    """Обработка сигнала входа в позицию - RAILWAY PRODUCTION MAINNET"""
    try:
        qty = float(data.get("qty", 0))
        
        # Обработка entry_price - может быть "market" или числом
        entry_price_raw = data.get("entry_price")
        if isinstance(entry_price_raw, str) and entry_price_raw.lower() == "market":
            entry_price = None  # Рыночный ордер
            logger.info(f"RAILWAY PRODUCTION MARKET ORDER: {symbol} {direction.upper()} qty={qty}")
        else:
            entry_price = float(entry_price_raw) if entry_price_raw else 0
            logger.info(f"RAILWAY PRODUCTION LIMIT ORDER: {symbol} {direction.upper()} qty={qty} price={entry_price}")
        
        # Обработка стоп-лосса и тейк-профита
        stop_loss = float(data.get("stop_loss")) if data.get("stop_loss") else None
        take_profit = float(data.get("take_profit")) if data.get("take_profit") else None
        leverage = 30  # Фиксированное плечо

        # Проверки безопасности
        if qty <= 0:
            return {"success": False, "error": "Количество должно быть больше 0"}
        
        # Проверка разумности стоп-лосса (только для лимитных ордеров)
        if entry_price and stop_loss:
            if direction.lower() == "long" and stop_loss >= entry_price:
                return {"success": False, "error": "Некорректный стоп-лосс для лонга"}
            elif direction.lower() == "short" and stop_loss <= entry_price:
                return {"success": False, "error": "Некорректный стоп-лосс для шорта"}

        logger.info(f"RAILWAY PRODUCTION ВХОД В ПОЗИЦИЮ: {symbol} {direction.upper()}")
        price_info = "MARKET" if entry_price is None else f"{entry_price}"
        logger.info(f"Количество: {qty}, Цена: {price_info}, SL: {stop_loss}, TP: {take_profit}")

        # Выполнение через V5 API только
        response = execute_trade_signal({
            "type": "entry",
            "symbol": symbol,
            "direction": direction,
            "qty": str(qty),
            "entry_price": "market" if entry_price is None else str(entry_price),
            "stop_loss": str(stop_loss) if stop_loss else None,
            "take_profit": str(take_profit) if take_profit else None,
            "leverage": leverage
        })

        if response.get("status") == "success":
            trade_stats["active_positions"] = trade_stats.get("active_positions", 0) + 1
            return {"success": True, "message": response.get("message"), "response": response}
        else:
            return {"success": False, "error": response.get("message")}

    except Exception as e:
        return {"success": False, "error": f"RAILWAY ошибка входа: {str(e)}"}

# handle_exit_signal функция удалена - используется execute_trade_signal из bybit_v5_fixed

def handle_trailing_signal(data, symbol, direction):
    """Обработка сигнала трейлинг стоп-лосса"""
    try:
        stop_price = float(data.get("stop_price", 0))
        trail_amount = float(data.get("trail_amount", 0))
        
        if stop_price <= 0:
            return {"success": False, "error": "Некорректный стоп-прайс"}
        
        logger.info(f"RAILWAY PRODUCTION ТРЕЙЛИНГ SL: {symbol} → {stop_price}")
        
        # Трейлинг через V5 API (пока простое обновление SL)
        logger.warning("RAILWAY: Трейлинг стоп пока не поддерживается в V5 API")
        return {"success": True, "message": f"RAILWAY: Трейлинг стоп получен (пока не реализован)", "response": {"trail_amount": trail_amount}}

    except Exception as e:
        return {"success": False, "error": f"RAILWAY ошибка трейлинга: {str(e)}"}

def handle_manual_close(data, symbol, direction):
    """Обработка ручного закрытия позиции"""
    try:
        logger.info(f"RAILWAY PRODUCTION РУЧНОЕ ЗАКРЫТИЕ: {symbol}")
        
        # Ручное закрытие через V5 API
        close_signal = {
            "symbol": symbol,
            "direction": "short" if direction.lower() == "long" else "long",
            "qty": 0.001,
            "entry_price": "market"
        }
        
        response = execute_trade_signal(close_signal)
        
        if response.get("status") == "success":
            trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 1) - 1)
            return {"success": True, "message": f"RAILWAY: Позиция {symbol} ручное закрытие", "response": response}
        else:
            return {"success": False, "error": response.get("message")}

    except Exception as e:
        return {"success": False, "error": f"RAILWAY ошибка ручного закрытия: {str(e)}"}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)