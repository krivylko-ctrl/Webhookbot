# 🚀 Быстрое развертывание на Render

## 1. Загрузить проект
- Распакуйте `mexc-trading-bot-latest.zip`
- Загрузите в новый Web Service на Render

## 2. Настройки Render
**Build Command:**
```
pip install -r requirements.txt
```

**Start Command:**
```
gunicorn --bind 0.0.0.0:$PORT --reuse-port main:app
```

**Environment:** Python 3

## 3. Переменные окружения
Добавить в Environment Variables:
```
MEXC_API_KEY = ваш_новый_api_ключ_от_mexc
MEXC_API_SECRET = ваш_новый_secret_от_mexc
SESSION_SECRET = любая_случайная_строка_32_символа
```

## 4. Проверка работы
После развертывания:
1. Откройте ваш сайт
2. Нажмите "Тест API" - должно показать успешное подключение
3. Скопируйте webhook URL: `https://ваш-домен.onrender.com/webhook`

## 5. Настройка TradingView
Используйте webhook URL в алертах Pine Script стратегии

## ✅ Готово!
Бот работает 24/7 автоматически