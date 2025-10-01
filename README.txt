TradingView to Bybit Webhook Bridge
====================================

Простой webhook-сервер для автоматической торговли на Bybit через сигналы от TradingView.

УСТАНОВКА И ЗАПУСК
==================

ЛОКАЛЬНЫЙ ЗАПУСК (для разработки):
-----------------------------------
1. Установите Python 3.11 или выше

2. Установите зависимости:
   pip install -r requirements.txt

3. Настройте переменные окружения:
   - Скопируйте .env.example в .env
   - Укажите ваши API ключи от Bybit:
     BYBIT_API_KEY=ваш_ключ
     BYBIT_API_SECRET=ваш_секрет
     BYBIT_TESTNET=false  (или true для тестовой сети)

4. Запустите сервер:
   python app.py

5. Сервер будет доступен на http://0.0.0.0:5000

DEPLOYMENT НА RAILWAY (для production):
---------------------------------------
1. Зарегистрируйтесь на railway.app

2. Создайте новый проект:
   - Нажмите "New Project"
   - Выберите "Deploy from GitHub repo" (или загрузите файлы напрямую)
   
3. Настройте переменные окружения в Railway:
   В разделе Variables добавьте:
   - BYBIT_API_KEY=ваш_ключ
   - BYBIT_API_SECRET=ваш_секрет
   - BYBIT_TESTNET=false
   - PORT=5000 (Railway автоматически установит свой порт)

4. Railway автоматически:
   - Обнаружит requirements.txt
   - Установит зависимости
   - Запустит через gunicorn (из Procfile)
   
5. После деплоя вы получите URL вида:
   https://web-production-xxxxx.up.railway.app

6. Используйте этот URL в TradingView:
   https://ваш-домен.up.railway.app/webhook


НАСТРОЙКА TRADINGVIEW
=====================

1. В TradingView создайте alert на вашей стратегии
2. В настройках алерта укажите:
   - Webhook URL: http://ваш_домен:5000/webhook
   - Message: оставьте {{strategy.order.alert_message}}

Стратегия автоматически отправит JSON в формате:
{
  "event": "open_block",
  "symbol": "ETHUSDT",
  "side": "Buy",
  "legs": [
    {
      "id": "05",
      "orderLinkId": "Buy_12345_1234567890_05",
      "price": "2000.5",
      "qty": "0.1",
      "lev": "25",
      "tp": "2100.0",
      "sl": "1950.0"
    }
  ],
  "oid_prefix": "Buy_12345_1234567890"
}


СТРУКТУРА ФАЙЛОВ
================

app.py              - Основной Flask сервер с webhook endpoints
bybit_client.py     - Клиент для работы с Bybit API v5
requirements.txt    - Список зависимостей Python
.env.example        - Пример файла с переменными окружения
.gitignore          - Файлы для исключения из git


ENDPOINTS
=========

GET  /              - Статус сервера
GET  /health        - Health check
POST /webhook       - Основной endpoint для приёма сигналов от TradingView


ФОРМАТ СИГНАЛОВ
===============

ОТКРЫТИЕ БЛОКА ОРДЕРОВ (open_block):
{
  "event": "open_block",
  "exchange": "bybit",
  "category": "linear",
  "symbol": "ETHUSDT",
  "side": "Buy" или "Sell",
  "legs": [
    {
      "id": "05",
      "orderLinkId": "уникальный_id",
      "price": "цена входа",
      "qty": "объём",
      "lev": "плечо (25, 30, 50)",
      "tp": "take profit",
      "sl": "stop loss"
    }
  ],
  "oid_prefix": "префикс для отмены"
}

ОТМЕНА БЛОКА ОРДЕРОВ (cancel_block):
{
  "event": "cancel_block",
  "exchange": "bybit",
  "category": "linear",
  "symbol": "ETHUSDT",
  "oid_prefix": "префикс_ордеров_для_отмены"
}


ЛОГИ
====

Все операции логируются в файл webhook.log
Формат: временная метка - модуль - уровень - сообщение


БЕЗОПАСНОСТЬ
============

⚠️ ВАЖНО:
- Никогда не делитесь вашими API ключами
- Используйте HTTPS для production
- Рекомендуется начать с Bybit Testnet
- API ключ должен иметь только права на Trade
- Используйте IP whitelist в настройках API на Bybit


ПОДДЕРЖКА
=========

При возникновении ошибок проверьте:
1. Правильность API ключей
2. Разрешения API ключа (Trade permissions)
3. Логи в файле webhook.log
4. Баланс на аккаунте Bybit
5. Корректность формата JSON от TradingView
