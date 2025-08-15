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

# Trail stats для монотонности и антиспама
trail_stats = {}

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
    """Enhanced TradingView webhook with new trail engine integration"""
    global trade_stats
    
    try:
        # Получение и обработка данных webhook
        raw_data = request.get_data(as_text=True)
        content_type = request.content_type or 'unknown'
        
        logger.info(f"TV WEBHOOK получен - Content-Type: {content_type}")
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
                logger.info(f"TV нормализация символа: {original_symbol} -> {data['symbol']}")
            logger.info(f"TV успешно распарсен JSON: {json.dumps(data, ensure_ascii=False, indent=2)}")
            
            # Фильтр точных дублей по отпечатку payload на 60 сек
            if is_duplicate_exact(data):
                msg = "Дубликат идентичного сигнала за 60с — игнор"
                return jsonify({"status":"blocked","message":msg}), 200
            
            add_log("INFO", f"Получен сигнал: {data.get('type', 'unknown')}")
        except json.JSONDecodeError as e:
            logger.error(f"TV невалидный JSON: {e}")
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

        # Обновление статистики
        trade_stats["total_signals"] += 1
        trade_stats["last_signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        trade_stats["last_signal_type"] = data['type']
        
        # Безопасный парсинг полей с дефолтами
        signal_type = (data.get("type") or "").lower()
        symbol = data.get("symbol", "ETHUSDT")
        direction = data.get("direction", "").lower()
        
        # Защита от дубликатов по 15-мин барам для entry
        if signal_type == "entry":
            bar_time = int(data.get("bar_time", 0))
            bar_key = (symbol, direction)
            if last_bar_entries.get(bar_key) == bar_time and bar_time > 0:
                msg = f"Duplicate entry for same 15m bar: {symbol} {direction}"
                add_log("WARNING", msg)
                return jsonify({"status": "blocked", "message": msg}), 200
            last_bar_entries[bar_key] = bar_time
        
        # Интеграция с новым TradingView Trail Engine
        try:
            from trail_engine_tv import (handle_entry, handle_trail_init, 
                                       handle_trail_update, handle_exit)
            
            if signal_type == "entry":
                add_log("INFO", f"🎯 Processing entry: {symbol} {direction}")
                
                entry_price = float(data.get("ref_price") or data.get("entry_price") or 0)
                stop_loss = float(data.get("stop_loss")) if data.get("stop_loss") is not None else None
                qty = float(data.get("qty")) if data.get("qty") is not None else None
                cancel_tp = bool(data.get("cancel_take_profit", False))
                
                # Валидация обязательных полей
                if not stop_loss or not qty:
                    return jsonify({"error": "Missing stop_loss or qty for entry"}), 400
                
                # Обработка через новый trail engine
                handle_entry(symbol, direction, entry_price, stop_loss, qty, cancel_take_profit=cancel_tp)
                
                # Выполняем вход через Bybit API
                result = execute_trade_signal({
                    "type": "entry",
                    "symbol": symbol,
                    "direction": direction,
                    "qty": str(qty),
                    "entry_price": "market" if not entry_price else str(entry_price),
                    "stop_loss": str(stop_loss),
                    "take_profit": str(data.get("take_profit")) if data.get("take_profit") and not cancel_tp else None
                })
                
                if result.get("status") == "success":
                    trade_stats["successful_trades"] += 1
                    trade_stats["active_positions"] = trade_stats.get("active_positions", 0) + 1
                    add_log("SUCCESS", f"✅ Entry executed: {symbol} {direction}")
                    return jsonify({"ok": True, "action": "entry"}), 200
                else:
                    trade_stats["failed_trades"] += 1
                    add_log("ERROR", f"❌ Entry failed: {result.get('message')}")
                    return jsonify({"error": result.get("message")}), 500
                
            elif signal_type == "trail_init":
                add_log("INFO", f"🔄 Processing trail init: {symbol} {direction}")
                
                hint_price = data.get("hint_price")
                hint_price = float(hint_price) if hint_price is not None else None
                trail_points = data.get("trail_points") 
                trail_points = float(trail_points) if trail_points is not None else None
                trail_offset = data.get("trail_offset")
                trail_offset = float(trail_offset) if trail_offset is not None else None
                force = bool(data.get("force", False))
                
                handle_trail_init(symbol, direction, hint_price, trail_points, trail_offset, force=force)
                add_log("SUCCESS", f"✅ Trail init: {symbol} {direction}")
                
                return jsonify({"ok": True, "action": "trail_init"}), 200
                
            elif signal_type == "trail_update":
                add_log("INFO", f"🔄 Processing trail update: {symbol} {direction}")
                
                new_trail_stop = float(data.get("new_trail_stop"))
                force = bool(data.get("force", False))
                
                handle_trail_update(symbol, direction, new_trail_stop, force=force)
                add_log("SUCCESS", f"✅ Trail update: {symbol} -> {new_trail_stop}")
                
                return jsonify({"ok": True, "action": "trail_update"}), 200
                
            elif signal_type == "exit":
                add_log("INFO", f"🚪 Processing exit: {symbol} {direction}")
                
                reason = data.get("reason", "")
                handle_exit(symbol, direction, reason)
                
                # Закрываем позицию на бирже
                result = execute_trade_signal({
                    "type": "exit",
                    "symbol": symbol,
                    "direction": direction,
                    "reason": reason
                })
                
                if result.get("status") == "success":
                    trade_stats["successful_trades"] += 1
                    trade_stats["active_positions"] = max(0, trade_stats.get("active_positions", 1) - 1)
                    add_log("SUCCESS", f"✅ Exit executed: {symbol} {direction}")
                else:
                    trade_stats["failed_trades"] += 1
                    add_log("ERROR", f"❌ Exit failed: {result.get('message')}")
                
                return jsonify({"ok": True, "action": "exit"}), 200
            
            else:
                add_log("WARNING", f"Unknown signal type: {signal_type}")
                return jsonify({"ok": False, "error": "unknown type"}), 400
                
        except Exception as e:
            trade_stats["failed_trades"] += 1
            error_msg = f"TV: Signal processing error: {str(e)}"
            logger.error(f"{error_msg}\n{traceback.format_exc()}")
            add_log("ERROR", error_msg)
            return jsonify({"ok": False, "error": str(e)}), 500
            
    except Exception as e:
        error_msg = f"TV: Critical webhook error: {str(e)}"
        logger.error(f"{error_msg}\n{traceback.format_exc()}")
        add_log("CRITICAL", error_msg)
        return jsonify({"ok": False, "error": str(e)}), 500

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
    """Обработка трейлинг сигнала - ТОЛЬКО модификация стоп-ордера, НЕ market"""
    try:
        from bybit_v5_fixed import get_position, v5_client as client
        
        # Извлечение данных + обработка Pine Script параметров
        new_trail_stop = float(data.get("new_trail_stop") or data.get("stop_price", 0))
        force = data.get("force", False)
        cancel_tp = data.get("cancel_take_profit", False)
        signal_id = f"{symbol}_{direction}_{round(new_trail_stop, 2)}"
        
        if new_trail_stop <= 0:
            return {"success": False, "error": "Некорректный стоп-прайс"}
        
        logger.info(f"TRAIL UPDATE: {symbol} {direction} → {new_trail_stop}")
        
        # 1) Получаем текущую позицию
        position = get_position(symbol)
        if not position or position.get('size', '0') == '0':
            return {"success": False, "error": "Нет открытой позиции для трейлинга"}
        
        pos_side = position.get('side', '').lower()
        pos_size = float(position.get('size', '0'))
        
        # 2) Защита от несоответствия стороны
        if pos_side != direction.lower():
            return {"success": False, "error": f"Position side mismatch: pos={pos_side} vs signal={direction}"}
        
        # 3) Получаем текущую рыночную цену
        try:
            ticker_response = client.get_tickers(category="linear", symbol=symbol)
            if ticker_response.get("retCode") != 0:
                return {"success": False, "error": "Failed to get market price"}
            
            mark_price = float(ticker_response.get("result", {}).get("list", [{}])[0].get("markPrice", 0))
            if mark_price <= 0:
                return {"success": False, "error": "Invalid mark price"}
                
        except Exception as e:
            return {"success": False, "error": f"Error getting market price: {e}"}
        
        # 4) Валидность цены стопа относительно рынка
        if pos_side == 'long' and new_trail_stop >= mark_price:
            return {"success": False, "error": f"Invalid stop for long: {new_trail_stop} >= {mark_price}"}
        if pos_side == 'short' and new_trail_stop <= mark_price:
            return {"success": False, "error": f"Invalid stop for short: {new_trail_stop} <= {mark_price}"}
        
        # 5) Проверка монотонности - загружаем последний стоп из кэша
        last_stop_key = f"last_stop_{symbol}_{pos_side}"
        last_stop = trail_stats.get(last_stop_key)
        
        if last_stop is not None:
            if pos_side == 'long' and new_trail_stop <= last_stop:
                return {"success": False, "error": f"Non-increasing stop: {new_trail_stop} <= {last_stop}"}
            if pos_side == 'short' and new_trail_stop >= last_stop:
                return {"success": False, "error": f"Non-decreasing stop: {new_trail_stop} >= {last_stop}"}
        
        # 6) Антиспам проверка (синхронизировано с Pine Script)
        now = time.time()
        last_update_key = f"last_update_{symbol}_{pos_side}"
        last_update_time = trail_stats.get(last_update_key, 0)
        
        # Pine Script: minAlertGapSec = 15, можем форсировать через force=true
        min_gap = 2.0 if force else 15.0
        if (now - last_update_time) < min_gap:
            return {"success": False, "error": f"Trail update too frequent (min gap: {min_gap}s)"}
        
        # 7) Минимальный шаг (синхронизировано с Pine Script baseTrailStepTicks = 5)
        try:
            # Получить размер тика из instruments info
            instruments_response = client.make_request("GET", "/v5/market/instruments-info", 
                                                     {"category": "linear", "symbol": symbol})
            if instruments_response.get("retCode") == 0:
                info = instruments_response.get("result", {}).get("list", [{}])[0]
                tick_size = float(info.get("priceFilter", {}).get("tickSize", "0.01"))
                min_step_ticks = 5  # как в Pine Script
                min_step_price = min_step_ticks * tick_size
                
                if last_stop is not None:
                    step_diff = abs(new_trail_stop - last_stop)
                    if not force and step_diff < min_step_price:
                        return {"success": False, "error": f"Step too small: {step_diff} < {min_step_price}"}
        except Exception as e:
            logger.warning(f"Could not get tick size for {symbol}: {e}")
        
        # 8) Обработка cancel_take_profit (как требует Pine Script)
        if cancel_tp:
            try:
                client.cancel_all_orders(category="linear", symbol=symbol, orderTypes="TakeProfit")
                logger.info(f"✅ TP orders cancelled for {symbol} as requested")
            except Exception as e:
                logger.warning(f"Failed to cancel TP orders: {e}")
        
        # 9) Создаем/обновляем триггерный стоп-ордер (НЕ market)
        side_for_close = "Sell" if pos_side == 'long' else "Buy"
        
        payload = {
            "category": "linear",
            "symbol": symbol,
            "side": side_for_close,
            "orderType": "Market",  # Market order при триггере
            "qty": str(abs(pos_size)),
            "triggerPrice": str(new_trail_stop),
            "triggerDirection": 1 if pos_side == 'long' else 2,  # 1=rising for long SL, 2=falling for short SL
            "triggerBy": "LastPrice",
            "reduceOnly": True,
            "timeInForce": "GTC",  # НЕ IOC
            "orderLinkId": f"trail_{symbol}_{pos_side}_{int(now)}"
        }
        
        # Сначала отменяем старые условные ордера
        try:
            client.cancel_all_orders(category="linear", symbol=symbol, settleCoin="USDT", 
                                   orderFilter="StopOrder")
            logger.info(f"Cancelled old stop orders for {symbol}")
        except:
            pass  # не критично
        
        # Размещаем новый условный ордер
        result = client.place_order(**payload)
        
        if result.get("retCode") == 0:
            # Сохраняем в кэше для монотонности
            trail_stats[last_stop_key] = new_trail_stop
            trail_stats[last_update_key] = now
            
            logger.info(f"✅ TRAIL STOP SET: {symbol} {pos_side} @ {new_trail_stop} (was {last_stop})")
            return {
                "success": True,
                "message": f"Trail stop updated to {new_trail_stop}",
                "price": new_trail_stop,
                "force_used": force,
                "cancel_tp_used": cancel_tp
            }
        else:
            error_msg = result.get("retMsg", "Unknown error placing stop order")
            return {"success": False, "error": f"Failed to place stop order: {error_msg}"}
            
    except Exception as e:
        return {"success": False, "error": f"Trail handler error: {str(e)}"}

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

@app.route('/tv-webhook', methods=['POST'])
def tv_webhook():
    """TradingView Enhanced Webhook - New Trail Engine Integration"""
    try:
        payload = request.get_json(force=True, silent=True) or {}
        signal_type = (payload.get("type") or "").lower()
        symbol = payload.get("symbol") or "UNKNOWN"
        direction = (payload.get("direction") or "").lower()

        logger.info(f"[TV-WEBHOOK] {signal_type}: {symbol} {direction}")
        add_log("INFO", f"TV Enhanced: {signal_type} {symbol} {direction}")

        from trail_engine_tv import (handle_entry, handle_trail_init, 
                                   handle_trail_update, handle_exit)

        if signal_type == "entry":
            entry_price = float(payload.get("ref_price") or payload.get("entry_price") or 0)
            stop_loss   = float(payload.get("stop_loss"))
            qty         = float(payload.get("qty") or 0)
            cancel_tp   = bool(payload.get("cancel_take_profit") or False)
            handle_entry(symbol, direction, entry_price, stop_loss, qty, cancel_take_profit=cancel_tp)
            return jsonify(ok=True, action="entry")

        elif signal_type == "trail_init":
            hint_price   = payload.get("hint_price")
            hint_price   = float(hint_price) if hint_price is not None else None
            trail_points = payload.get("trail_points")
            trail_points = float(trail_points) if trail_points is not None else None
            trail_offset = payload.get("trail_offset")
            trail_offset = float(trail_offset) if trail_offset is not None else None
            force        = bool(payload.get("force") or False)
            handle_trail_init(symbol, direction, hint_price, trail_points, trail_offset, force=force)
            return jsonify(ok=True, action="trail_init")

        elif signal_type == "trail_update":
            new_trail_stop = float(payload.get("new_trail_stop"))
            force          = bool(payload.get("force") or False)
            handle_trail_update(symbol, direction, new_trail_stop, force=force)
            return jsonify(ok=True, action="trail_update")

        elif signal_type == "exit":
            reason = payload.get("reason", "")
            handle_exit(symbol, direction, reason)
            logger.info(f"[EXIT] {symbol} {direction} reason={reason}")
            return jsonify(ok=True, action="exit")

        else:
            return jsonify(ok=False, error="unknown type"), 400

    except Exception as e:
        logger.exception("tv-webhook error")
        return jsonify(ok=False, error=str(e)), 500

if __name__ == '__main__':
    try:
        # Запуск trail engine в фоновом режиме
        trail_engine.start()
        logger.info("🎯 Trail engine started successfully")
        
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise