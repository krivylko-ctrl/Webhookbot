# 🚀 Railway Deployment Instructions

## Быстрое развертывание на Railway

### 1. Подготовка проекта
```bash
# Создайте новый репозиторий на GitHub и загрузите файлы из папки railway_deploy/
```

### 2. Настройка Railway
1. Перейдите на [railway.app](https://railway.app)
2. Войдите через GitHub
3. Нажмите "New Project" → "Deploy from GitHub repo"
4. Выберите ваш репозиторий
5. Railway автоматически определит Python проект

### 3. Переменные окружения
Добавьте в Railway следующие переменные:

```bash
# Основные API ключи
BYBIT_API_KEY=ваш_bybit_api_key
BYBIT_API_SECRET=ваш_bybit_secret

# Безопасность
SESSION_SECRET=railway-prod-secret-2025

# Опционально для Telegram бота
TELEGRAM_BOT_TOKEN=ваш_telegram_token
TELEGRAM_CHAT_ID=ваш_chat_id
```

### 4. Настройка домена
После развертывания Railway предоставит:
- **Публичный URL**: `https://your-project-name.up.railway.app`
- **Webhook URL**: `https://your-project-name.up.railway.app/webhook`

### 5. Настройка TradingView
В вашей Pine Script стратегии добавьте:

```pinescript
strategy.entry("Long", strategy.long, qty=position_size, 
    alert_message='{"type":"entry","symbol":"ETHUSDT","direction":"long","qty":' + str.tostring(position_size) + ',"leverage":10,"sl_percent":1.0,"tp_percent":3.0}')

strategy.close("Long", 
    alert_message='{"type":"exit","symbol":"ETHUSDT","direction":"long"}')
```

**Webhook URL в TradingView**: `https://your-project-name.up.railway.app/webhook`

### 6. Проверка развертывания

1. **Health Check**: `GET https://your-project-name.up.railway.app/health`
2. **Dashboard**: `https://your-project-name.up.railway.app/`
3. **Test API**: Кнопка "Тест API" в интерфейсе

### 7. Мониторинг

Railway предоставляет:
- **Логи в реальном времени**
- **Метрики использования**
- **Автоматические перезапуски**
- **SSL сертификаты**

### 8. Преимущества Railway

✅ **Лучшая совместимость с международными API**
✅ **Отсутствие CloudFront блокировок**
✅ **Автоматический SSL и домены**
✅ **Встроенный мониторинг**
✅ **Простое управление переменными**
✅ **Git-based развертывание**

### 9. Режимы работы

**PRODUCTION MODE** (по умолчанию):
- Реальная торговля на основном аккаунте Bybit
- Все ордера выполняются на бирже
- testnet=False

**DEMO MODE** (при ошибках API):
- Автоматическое переключение на симулятор
- Логирование всех операций
- Безопасное тестирование

### 10. Troubleshooting

**API Ошибки**:
- Проверьте API ключи в переменных окружения
- Убедитесь что IP разрешен в Bybit
- Система автоматически переключится на симулятор

**Webhook Ошибки**:
- Проверьте JSON формат в TradingView
- URL должен быть HTTPS
- Время ответа должно быть < 5 секунд

**Логи для отладки**:
```bash
# В Railway Dashboard → Deployments → View Logs
```

---

## 🎯 Готовые файлы включены:

- ✅ `app.py` - Основное приложение Flask
- ✅ `bybit_api.py` - CCXT интеграция с Bybit
- ✅ `bybit_simulator.py` - Симулятор торговли
- ✅ `requirements.txt` - Python зависимости
- ✅ `Procfile` - Конфигурация для Railway
- ✅ `main.py` - Точка входа
- ✅ `templates/` - HTML интерфейс
- ✅ `runtime.txt` - Версия Python

**Система готова к production развертыванию!**