# 🚀 KWIN Trading Bot - Развертывание на Railway

## Описание
KWIN Trading Bot - автономный торговый бот для криптовалют с реализацией стратегии Swing Failure Pattern (SFP), Smart Trailing и полным веб-интерфейсом на Streamlit.

## 📋 Требования
- Аккаунт на [Railway.app](https://railway.app)
- Аккаунт на [GitHub](https://github.com)
- API ключи от Bybit

## 🔧 Инструкция по установке

### Шаг 1: Подготовка репозитория GitHub

1. **Создайте новый репозиторий на GitHub:**
   - Перейдите на GitHub.com
   - Нажмите "New repository"
   - Название: `kwin-trading-bot`
   - Сделайте репозиторий публичным или приватным

2. **Загрузите файлы проекта:**
   - Скачайте все файлы из текущего проекта
   - Загрузите их в свой GitHub репозиторий
   - Убедитесь что есть все основные файлы:
     ```
     app.py
     bybit_api.py
     kwin_strategy.py
     trail_engine.py
     state_manager.py
     config.py
     database.py
     utils.py
     demo_mode.py
     pages/
     .streamlit/config.toml
     railway_deployment_requirements.txt (переименуйте в requirements.txt)
     Procfile
     railway.json
     ```

### Шаг 2: Развертывание на Railway

1. **Создание проекта:**
   - Перейдите на [Railway.app](https://railway.app)
   - Войдите через GitHub
   - Нажмите "New Project"
   - Выберите "Deploy from GitHub repo"
   - Выберите ваш репозиторий `kwin-trading-bot`

2. **Настройка переменных окружения:**
   В разделе "Variables" добавьте:
   ```
   BYBIT_API_KEY=ваш_api_ключ
   BYBIT_API_SECRET=ваш_api_секрет
   PORT=5000
   ```

3. **Настройка домена:**
   - В разделе "Settings" найдите "Public Networking"
   - Нажмите "Generate Domain"
   - Сохраните полученный URL

### Шаг 3: Получение API ключей Bybit

1. **Вход в Bybit:**
   - Перейдите на [Bybit.com](https://www.bybit.com)
   - Войдите в свой аккаунт

2. **Создание API ключа:**
   - API Management → Create New Key
   - Название: "KWIN Trading Bot"
   - Разрешения:
     - ✅ Read-Write
     - ✅ Spot Trading
     - ✅ Wallet
   - IP Whitelist: оставьте пустым или добавьте IP Railway
   - Скопируйте API Key и Secret

3. **Настройка разрешений:**
   - Убедитесь что включены:
     - Contract Trading (если планируете фьючерсы)
     - Spot Trading (для спот торговли)
     - Wallet (для чтения баланса)

### Шаг 4: Проверка развертывания

1. **Проверка сборки:**
   - В Railway откройте раздел "Deployments"
   - Дождитесь завершения сборки (зеленая галочка)
   - При ошибках проверьте логи

2. **Тестирование приложения:**
   - Откройте сгенерированный URL
   - Проверьте статус подключения к API
   - Убедитесь что интерфейс загружается

## 🎯 Структура проекта

```
kwin-trading-bot/
├── app.py                 # Главная страница приложения
├── pages/                 # Дополнительные страницы
│   ├── 1_Dashboard.py     # Дашборд с метриками
│   ├── 2_Backtest.py      # Страница бэктестинга
│   └── 3_Settings.py      # Настройки бота
├── bybit_api.py          # API интеграция с Bybit
├── kwin_strategy.py      # Основная торговая стратегия
├── trail_engine.py       # Система трейлинга
├── state_manager.py      # Управление состоянием
├── config.py            # Конфигурация параметров
├── database.py          # База данных SQLite
├── utils.py             # Вспомогательные функции
├── demo_mode.py         # Демо режим для тестирования
├── requirements.txt     # Python зависимости
├── Procfile            # Команда запуска для Railway
├── railway.json        # Конфигурация Railway
└── .streamlit/         # Настройки Streamlit
    └── config.toml
```

## ⚙️ Основные функции

- **SFP Detection**: Автоматическое обнаружение Swing Failure Patterns
- **Smart Trailing**: Умная система трейлинг-стопов
- **Risk Management**: Управление рисками с фиксированным процентом
- **Real-time Trading**: Торговля в реальном времени через Bybit API
- **Web Interface**: Полный веб-интерфейс для мониторинга
- **Backtesting**: Тестирование стратегии на исторических данных
- **Database**: Сохранение истории сделок и статистики

## 🔒 Безопасность

- API ключи хранятся в переменных окружения
- Никогда не коммитьте API ключи в репозиторий
- Используйте IP whitelist для API ключей
- Регулярно проверяйте активность API ключей

## 📞 Поддержка

При возникновении проблем:
1. Проверьте логи Railway в разделе "Deployments"
2. Убедитесь в правильности API ключей
3. Проверьте статус подключения к Bybit
4. Убедитесь что все файлы загружены в репозиторий

## 🚨 Важные замечания

- Тестируйте стратегию перед реальной торговлей
- Начинайте с небольших сумм
- Регулярно мониторьте работу бота
- Сохраняйте резервные копии настроек