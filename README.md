# 🚀 Railway Bybit Production Trading Bot

**Готовая система для автоматической торговли ETH с SFP стратегией на Bybit mainnet через Railway deployment**

## ⚡ Быстрый запуск на Railway

### 1. Создание проекта на Railway
1. Перейдите на [railway.app](https://railway.app)
2. Создайте новый проект
3. Выберите "Deploy from GitHub repo"
4. Загрузите все файлы из папки `railway_production_ready/`

### 2. Настройка переменных окружения
В настройках Railway проекта добавьте:

```bash
BYBIT_API_KEY=ваш_production_api_key
BYBIT_API_SECRET=ваш_production_api_secret
SESSION_SECRET=railway-production-secret-2025
PORT=5000
```

⚠️ **ВАЖНО**: Используйте только production API ключи от основного аккаунта Bybit mainnet!

### 3. Deployment
Railway автоматически развернет приложение после:
- Добавления переменных окружения
- Загрузки файлов проекта
- Активации deployment

### 4. Проверка работы
1. Откройте Railway dashboard URL
2. Проверьте статус "Healthy" в `/health`
3. Протестируйте соединение через веб-интерфейс
4. Настройте webhook URL в TradingView

## 🔧 Архитектура системы

### Основные файлы:
- `app.py` - Flask приложение с webhook endpoints
- `bybit_api.py` - Заглушка (CCXT удален)
- `requirements.txt` - Python зависимости
- `railway.toml` - Конфигурация Railway
- `templates/` - HTML интерфейс

### Особенности Railway версии:
- ✅ Обход CloudFront блокировки
- ✅ Обработка "market" price сигналов
- ✅ Production mainnet API
- ✅ Автоматический restart при ошибках
- ✅ Health check monitoring
- ✅ Логирование всех операций

## 📡 Webhook URL для TradingView

После deployment ваш webhook URL будет:
```
https://ваш-проект.railway.app/webhook
```

Используйте этот URL в настройках алертов TradingView Pine Script.

## 🔒 Безопасность

- Все API ключи хранятся в переменных окружения
- Production mainnet - реальные деньги
- Защита от дублирующих сигналов
- Экстренная остановка торговли
- Логирование всех операций

## 📊 Мониторинг

### Web Dashboard:
- Статистика сделок
- Активные позиции
- Состояние системы
- Последние события

### API Endpoints:
- `/health` - Health check
- `/api/stats` - Статистика
- `/api/test_connection` - Тест API
- `/webhook` - Прием сигналов TradingView

## ⚠️ Важные моменты

1. **Реальные деньги**: Система работает на production mainnet
2. **API ключи**: Требуются production ключи от основного аккаунта Bybit
3. **CloudFront**: Railway обходит географические ограничения
4. **Объемы**: Точные объемы из сигналов сохраняются
5. **Leverage**: Фиксированное плечо 30x

## 🆘 Поддержка

При проблемах проверьте:
1. Railway logs в dashboard
2. Health check status
3. API ключи в переменных окружения
4. Webhook URL в TradingView

## 📈 Стратегия

Система работает с SFP (Swing Failure Pattern) стратегией:
- Таймфрейм: 15 минут
- Актив: ETHUSDT
- Рыночные ордера ("market" price)
- Автоматические стоп-лоссы
- Трейлинг стопы

**Система готова к боевой торговле на Railway! 🚀**