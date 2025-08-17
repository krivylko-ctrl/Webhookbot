# KWIN Trading Bot

## Overview

KWIN Trading Bot v2.0 is an advanced cryptocurrency trading automation system implementing Swing Failure Pattern (SFP) detection with comprehensive analytics and smart trailing. The system features enhanced SFP quality validation, detailed trail logging, unit/e2e testing suite, and a complete analytics dashboard. Built with Python and Streamlit, it provides professional-grade trading automation with 100% Pine Script compatibility and robust risk management.

## Recent Changes (August 2025)

### Version 2.2.0-Ultimate (17 августа 2025)
- **ULTIMATE STRATEGY RELEASE** - kwin_strategy.py (переименован) с 99%+ Pine совместимостью
- **КРИТИЧЕСКИЕ ИСПРАВЛЕНИЯ** - Исправлены все импорты, зависимости и функциональность
- **ПОЛНОФУНКЦИОНАЛЬНЫЙ BACKTEST** - pages/2_Backtest.py полностью переписан (300+ строк)
- **WEBSOCKET ИСПРАВЛЕН** - Полнофункциональная интеграция с Bybit WebSocket API v5
- **ПОЛНАЯ АНАЛИТИКА** - pages/1_Analytics.py с комплексными графиками и метриками
- **Pine-like TA Functions** - точные ta.pivotlow/pivothigh для newest-first массивов
- **Series Accessor** - Pine-like _series(field, tf) эмулирующий request.security()
- **Close-back фикс** - нормализация в [0..1] диапазон как в оригинальном Pine
- **Smart Trailing улучшения** - правильная обработка экстремумов и trail_engine интеграция
- **Комиссии из конфига** - fee_rate берется из config.taker_fee_rate вместо хардкода
- **Database in-memory** - поддержка Database(memory=True) для бэктестов
- **LSP диагностика** - исправлены все синтаксические ошибки и warnings
- **Streamlit оптимизация** - использование кешированных ресурсов и эффективного рендеринга
- **WebSocket мониторинг** - Real-time данные BTCUSDT/ETHUSDT с правильной обработкой символов

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
The application uses **Streamlit** as the web framework, providing a multi-page dashboard interface. The frontend is organized into separate pages:
- **Main Dashboard** (`app.py`) - Overview and real-time monitoring
- **Dashboard** (`pages/1_Dashboard.py`) - Detailed trading metrics and charts
- **Backtest** (`pages/2_Backtest.py`) - Historical strategy performance analysis
- **Settings** (`pages/3_Settings.py`) - Configuration management

The UI leverages **Plotly** for interactive charts and visualizations, with automatic refresh capabilities for real-time data updates.

### Backend Architecture
The system follows a **modular component-based architecture** with clear separation of concerns:

- **Strategy Engine** (`kwin_strategy.py`) - Core trading logic implementing SFP detection and trade execution
- **State Manager** (`state_manager.py`) - Centralized state management for positions, equity, and bot status
- **Trail Engine** (`trail_engine.py`) - Smart trailing stop-loss implementation with multiple trailing modes
- **API Layer** (`bybit_api.py`) - Exchange API integration with authentication and rate limiting
- **Configuration** (`config.py`) - Centralized parameter management with file-based persistence
- **Utilities** (`utils.py`) - Common mathematical functions for price rounding, PnL calculations, and risk metrics

### Data Storage Solutions
The system uses **SQLite** as the primary database (`database.py`) with the following schema design:
- **Trades table** - Complete trade history with entry/exit prices, PnL, and metadata
- **Equity history table** - Timeline of account balance changes
- **Bot state table** - Persistent storage of current positions and system status

This approach provides lightweight, file-based persistence suitable for single-instance deployment while maintaining data integrity.

### Authentication and Authorization
**API-based authentication** is implemented for Bybit exchange integration:
- API credentials stored as environment variables (`BYBIT_API_KEY`, `BYBIT_API_SECRET`)
- HMAC-SHA256 signature generation for authenticated requests
- Support for both testnet and mainnet environments
- Session-based HTTP client with proper error handling

### Trading Strategy Logic
The **KWIN strategy** implements:
- **SFP Detection** - Identifies swing failure patterns using pivot analysis on 15m timeframes
- **Quality Filtering** - Validates SFP patterns based on wick depth and close-back percentage
- **Risk Management** - Position sizing based on fixed risk percentage with maximum quantity limits
- **Smart Trailing** - Multi-mode trailing stops including percentage-based and bar-based trailing
- **Arming Mechanism** - Conditional activation of trailing after reaching minimum risk/reward ratios

## External Dependencies

### Exchange Integration
- **Bybit API v5** - Primary exchange for live trading and market data
- **WebSocket streams** - Real-time price feeds and order updates
- **REST API** - Order management, account information, and historical data

### Python Libraries
- **Streamlit** - Web application framework for the user interface
- **Plotly** - Interactive charting and data visualization
- **Pandas/NumPy** - Data manipulation and numerical computations
- **Requests** - HTTP client for API communications
- **SQLite3** - Database operations and persistence

### Development Tools
- **Threading** - Concurrent processing for WebSocket connections and data updates
- **JSON** - Configuration serialization and API data handling
- **Datetime/Time** - Timestamp management and scheduling operations

### Environment Configuration
- **Environment Variables** - Secure storage of API credentials and deployment settings
- **File-based Configuration** - JSON persistence for strategy parameters and user preferences