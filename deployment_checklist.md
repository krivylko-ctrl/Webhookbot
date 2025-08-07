# Чек-лист развертывания на Render

## 📦 Готовый архив
`mexc-trading-bot-ready.zip` - содержит актуальную версию проекта

## 🔑 Переменные окружения (Environment Variables)
В настройках Render добавить:
```
MEXC_API_KEY=ваш_новый_api_ключ
MEXC_API_SECRET=ваш_новый_secret_ключ  
SESSION_SECRET=любая_случайная_строка_для_flask
```

## 🚀 Настройки развертывания
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT --reuse-port main:app`
- **Environment**: Python 3
- **Auto-Deploy**: Включить

## 📋 После развертывания
1. Проверить работу веб-интерфейса
2. Протестировать API подключение кнопкой "Тест API"
3. Скопировать webhook URL для TradingView
4. Настроить алерты в TradingView

## 🎯 Webhook URL
После развертывания будет: `https://ваш-домен.onrender.com/webhook`

## ⚠️ Важно
- Новые API ключи должны быть активированы для фьючерсной торговли
- Первый запуск может занять 1-2 минуты
- Webhook работает 24/7 автоматически