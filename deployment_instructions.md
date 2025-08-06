# Инструкция по развертыванию торгового бота на Render

## 📋 Требования
- Аккаунт на render.com
- Файлы проекта (скачаны из Replit)
- API ключи от MEXC

## 🚀 Пошаговое развертывание

### 1. Подготовка файлов
Убедитесь, что у вас есть все файлы:
- app.py
- mexc_api.py
- main.py
- templates/index.html
- templates/logs.html
- static/style.css
- requirements.txt
- README.md

### 2. Создание сервиса на Render

1. Зайдите на **render.com**
2. Нажмите **"New +"** → **"Web Service"**
3. Выберите **"Deploy from uploaded files"**
4. Загрузите ZIP архив с файлами проекта

### 3. Настройка сервиса

**Основные настройки:**
- **Name**: mexc-trading-bot
- **Region**: Oregon (US West)
- **Branch**: main
- **Runtime**: Python 3
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn --bind 0.0.0.0:$PORT main:app`

### 4. Переменные окружения

Добавьте в разделе **Environment Variables**:

```
MEXC_API_KEY = ваш_api_ключ_mexc
MEXC_API_SECRET = ваш_секретный_ключ_mexc
SESSION_SECRET = любая_случайная_строка_32_символа
```

**Важно:** Начните с testnet ключей для безопасности!

### 5. Развертывание

1. Нажмите **"Create Web Service"**
2. Дождитесь завершения сборки (5-10 минут)
3. Получите URL вашего приложения: `https://your-app-name.onrender.com`

## 🔧 Настройка TradingView

### 1. Создание алерта
1. Откройте график в TradingView
2. Настройте условия алерта
3. В поле **Webhook URL** укажите: `https://your-app-name.onrender.com/webhook`

### 2. Формат сообщения
Используйте JSON формат:
```json
{
  "type": "entry",
  "direction": "long",
  "qty": 1.5,
  "entry_price": 2500.0,
  "stop_loss": 2450.0,
  "take_profit": 2600.0,
  "symbol": "ETHUSDT"
}
```

## 📱 Использование с iPad

### Доступ к интерфейсу:
1. Откройте Safari или Chrome на iPad
2. Перейдите по URL: `https://your-app-name.onrender.com`
3. Используйте веб-интерфейс для мониторинга

### Возможности:
- ✅ Просмотр статистики торгов
- ✅ Мониторинг логов в реальном времени
- ✅ Тестирование подключения к MEXC
- ✅ Получение webhook URL для TradingView

## 🔐 Безопасность

### Рекомендации:
1. **Начните с testnet** для проверки
2. **Ограничьте права API ключей** в MEXC (только торговля)
3. **Не делитесь URL webhook** с третьими лицами
4. **Регулярно проверяйте логи** на подозрительную активность

### Переход на mainnet:
1. Создайте реальные API ключи в MEXC
2. Обновите переменные окружения в Render
3. Измените в mexc_api.py: `testnet=False`

## 🔄 Обновление кода

При необходимости изменений:
1. Загрузите новые файлы через Render dashboard
2. Сервис автоматически перезапустится
3. Проверьте логи на отсутствие ошибок

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи в Render dashboard
2. Убедитесь в корректности API ключей
3. Проверьте формат webhook сообщений
4. Используйте раздел логов в веб-интерфейсе

---
*Торговый бот готов к работе 24/7!*