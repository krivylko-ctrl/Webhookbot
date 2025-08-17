from flask import Flask, render_template, request, jsonify
import os
import json
from datetime import datetime
import threading

from bybit_api import BybitAPI
from kwin_strategy import KWINStrategy
from state_manager import StateManager
from database import Database
from config import Config

app = Flask(__name__)

# Глобальные переменные для компонентов
config = None
db = None
state_manager = None
bybit_api = None
strategy = None
logs = []

def init_components():
    """Инициализация компонентов"""
    global config, db, state_manager, bybit_api, strategy
    
    config = Config()
    db = Database()
    state_manager = StateManager(db)
    
    # Получаем API ключи из переменных окружения
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    
    if api_key and api_secret:
        bybit_api = BybitAPI(api_key, api_secret, testnet=False)
        try:
            server_time = bybit_api.get_server_time()
            if not server_time:
                from demo_mode import create_demo_api
                bybit_api = create_demo_api()
                log_message("⚠️ Bybit API недоступен. Включен демо-режим.")
        except Exception as e:
            from demo_mode import create_demo_api
            bybit_api = create_demo_api()
            log_message(f"⚠️ Ошибка Bybit API: {e}. Включен демо-режим.")
    else:
        from demo_mode import create_demo_api
        bybit_api = create_demo_api()
        log_message("ℹ️ API ключи не настроены. Работаем в демо-режиме.")
    
    strategy = KWINStrategy(config, bybit_api, state_manager, db)
    log_message("🚀 KWIN Trading Bot инициализирован")

def log_message(message):
    """Добавление сообщения в лог"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    logs.append(log_entry)
    # Ограничиваем размер лога
    if len(logs) > 1000:
        logs[:] = logs[-500:]
    print(log_entry)

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/logs')
def view_logs():
    """Страница с логами"""
    log_text = "\n".join(logs) if logs else "Логи пусты"
    return render_template('logs.html', logs=log_text)

@app.route('/health')
def health():
    """Проверка здоровья приложения"""
    return jsonify({
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "bot_initialized": strategy is not None,
        "api_connected": bybit_api is not None
    })

@app.route('/webhook', methods=['POST'])
def webhook():
    """Webhook для получения сигналов"""
    try:
        data = request.get_json()
        log_message(f"📥 Webhook получен: {data}")
        
        if not strategy:
            return jsonify({"error": "Bot not initialized"}), 500
        
        # Обработка webhook сигнала
        if data.get('action') == 'BUY':
            log_message("📈 Получен сигнал BUY")
            # Логика обработки покупки
        elif data.get('action') == 'SELL':
            log_message("📉 Получен сигнал SELL")
            # Логика обработки продажи
        
        return jsonify({"status": "success", "message": "Signal processed"})
    
    except Exception as e:
        log_message(f"❌ Ошибка webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/tv-webhook', methods=['POST'])
def tv_webhook():
    """TradingView webhook"""
    try:
        data = request.get_json()
        log_message(f"📊 TradingView webhook: {data}")
        
        if not strategy:
            return jsonify({"error": "Bot not initialized"}), 500
        
        # Обработка TradingView сигнала
        return jsonify({"status": "success", "message": "TradingView signal processed"})
    
    except Exception as e:
        log_message(f"❌ Ошибка TradingView webhook: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/status')
def status():
    """Статус бота"""
    if not strategy:
        return jsonify({"error": "Bot not initialized"}), 500
    
    try:
        current_position = state_manager.get_current_position()
        equity = state_manager.get_equity_history()
        
        return jsonify({
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "position": current_position,
            "equity": equity,
            "symbol": config.symbol if config else None,
            "logs_count": len(logs)
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Инициализация при запуске
    init_components()
    
    # Запуск Flask приложения
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)