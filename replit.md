# Overview

This is a Flask-based cryptocurrency trading bot that integrates with the MEXC exchange API to execute automated trades based on webhook signals. The application provides a web dashboard for monitoring trading statistics, managing positions, and viewing detailed logs. The bot is designed to receive trading signals from external sources (likely TradingView alerts) and automatically execute trades on MEXC futures markets using technical analysis strategies.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
The application uses a server-side rendered Flask web interface with Bootstrap 5 dark theme for responsive design. The frontend consists of two main pages:
- **Dashboard (`index.html`)** - Displays trading statistics, recent activity, and control buttons for API testing and data refresh
- **Logs page (`logs.html`)** - Provides detailed log viewing with filtering capabilities by level and search functionality

The UI uses Font Awesome icons and custom CSS for enhanced visual appeal. JavaScript handles client-side interactions like refreshing data and filtering logs.

## Backend Architecture
The backend follows a simple Flask application structure:
- **`app.py`** - Main Flask application with webhook endpoints, statistics tracking, and log management
- **`mexc_api.py`** - MEXC exchange API client with authentication, signature generation, and trading methods
- **`main.py`** - Application entry point that starts the Flask development server

The application uses in-memory storage for statistics and recent logs, with a maximum of 100 log entries retained. Trade statistics include total signals, successful/failed trades, last signal information, and active positions count.

## Trading Strategy Integration
The bot is designed to work with Pine Script trading strategies (evidenced by the TradingView strategy file). The strategy appears to use:
- Swing failure patterns (SFP) on 15-minute timeframes
- RSI-based position sizing and entry conditions
- Risk management with configurable stop-loss and take-profit ratios
- Support for both long and short positions with trailing stops

## API Integration Pattern
The MEXC API client implements:
- HMAC-SHA256 signature authentication
- Request timestamping for security
- Support for both testnet and mainnet environments
- Position opening/closing with leverage support
- Error handling and logging for all API interactions

## Logging and Monitoring
The application implements comprehensive logging:
- File-based logging to `webhook_log.txt` with UTF-8 encoding
- Console output for development
- In-memory recent logs storage for web dashboard
- Multiple log levels (INFO, WARNING, ERROR, CRITICAL, SUCCESS)
- Request/response logging for API calls

# External Dependencies

## MEXC Exchange API
- **Purpose**: Cryptocurrency futures trading execution
- **Authentication**: API key and secret with HMAC signature
- **Endpoints**: Position management, order placement, account information
- **Environment**: Configurable testnet/mainnet support

## Bootstrap 5 & Font Awesome
- **Purpose**: Frontend UI framework and iconography
- **Source**: CDN-hosted for responsive design and dark theme support

## Flask Framework
- **Purpose**: Web application framework for Python
- **Features**: Request handling, templating, session management
- **Extensions**: Built-in Jinja2 templating engine

## Python Standard Libraries
- **requests**: HTTP client for API communication
- **hmac/hashlib**: Cryptographic signature generation
- **logging**: Application logging and debugging
- **datetime**: Timestamp handling and time-based operations
- **json**: Data serialization for API communication

## TradingView Integration
- **Purpose**: Receives trading signals via webhooks
- **Format**: JSON payload with trade type, direction, symbol, and parameters
- **Strategy**: Pine Script-based technical analysis with SFP and RSI indicators