import requests
import time
import hmac
import hashlib
import uuid
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class MexcFuturesClient:
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = True):
        self.api_key = api_key or os.getenv("MEXC_API_KEY", "mx0vglvOO7WTZLIk3c")
        self.api_secret = api_secret or os.getenv("MEXC_API_SECRET", "d999f1140d744bf99103425508f44bb0")
        
        if testnet:
            self.base_url = "https://contract-test.mexc.com"
        else:
            self.base_url = "https://contract.mexc.com"
            
        logger.info(f"Инициализирован MEXC клиент для {'testnet' if testnet else 'mainnet'}")

    def generate_signature(self, req_time: str, sign_params_str: str) -> str:
        """Генерация подписи для MEXC API"""
        message = f"{self.api_key}{req_time}{sign_params_str}"
        return hmac.new(
            self.api_secret.encode(), 
            message.encode(), 
            hashlib.sha256
        ).hexdigest()

    def _make_request(self, endpoint: str, params: str) -> Dict[str, Any]:
        """Базовый метод для выполнения запросов к MEXC API"""
        try:
            req_time = str(int(time.time() * 1000))
            signature = self.generate_signature(req_time, params)
            
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "ApiKey": self.api_key,
                "Request-Time": req_time,
                "Signature": signature
            }
            
            url = f"{self.base_url}{endpoint}"
            logger.info(f"Отправка запроса к {url}")
            logger.debug(f"Параметры: {params}")
            
            response = requests.post(url, data=params, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Ответ API: {result}")
            return result
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка HTTP запроса: {e}")
            if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
                return {"error": "Нет подключения к интернету или MEXC API недоступен", "demo_mode": True}
            return {"error": f"HTTP ошибка: {str(e)}"}
        except Exception as e:
            logger.error(f"Неожиданная ошибка: {e}")
            return {"error": f"Неожиданная ошибка: {str(e)}"}

    def open_position(self, symbol: str, direction: str, quantity: float, 
                     entry_price: float, stop_loss: float, take_profit: float, 
                     leverage: int = 20) -> Dict[str, Any]:
        """
        Открытие позиции
        
        Args:
            symbol: Торговая пара (например, "ETHUSDT")
            direction: Направление ("long" или "short")
            quantity: Количество
            entry_price: Цена входа
            stop_loss: Стоп-лосс
            take_profit: Тейк-профит
            leverage: Плечо (по умолчанию 20)
        """
        try:
            side = 1 if direction.lower() == "long" else 2
            oid = str(uuid.uuid4())
            
            params = (
                f"symbol={symbol}&price=0&vol={quantity}&leverage={leverage}"
                f"&side={side}&type=1&open_type=1&position_id=0"
                f"&external_oid={oid}&stop_loss_price={stop_loss}"
                f"&take_profit_price={take_profit}&position_mode=1"
            )
            
            logger.info(f"Открытие позиции: {direction} {symbol}, кол-во: {quantity}, "
                       f"SL: {stop_loss}, TP: {take_profit}, плечо: {leverage}")
            
            return self._make_request("/api/v1/private/order/submit", params)
            
        except Exception as e:
            logger.error(f"Ошибка открытия позиции: {e}")
            return {"error": f"Ошибка открытия позиции: {str(e)}"}

    def close_position(self, symbol: str, direction: str, quantity: Optional[float] = None) -> Dict[str, Any]:
        """
        Закрытие позиции
        
        Args:
            symbol: Торговая пара
            direction: Направление исходной позиции ("long" или "short")
            quantity: Количество для закрытия (если None, закрывается вся позиция)
        """
        try:
            # Для закрытия используется противоположное направление
            side = 2 if direction.lower() == "long" else 1
            oid = str(uuid.uuid4())
            vol = quantity if quantity is not None else 0.01  # Минимальное значение для API
            
            params = (
                f"symbol={symbol}&price=0&vol={vol}&leverage=20"
                f"&side={side}&type=1&open_type=1&position_id=0"
                f"&external_oid={oid}&stop_loss_price=&take_profit_price=&position_mode=1"
            )
            
            logger.info(f"Закрытие позиции: {direction} {symbol}")
            
            return self._make_request("/api/v1/private/order/submit", params)
            
        except Exception as e:
            logger.error(f"Ошибка закрытия позиции: {e}")
            return {"error": f"Ошибка закрытия позиции: {str(e)}"}

    def edit_position(self, symbol: str, direction: str, stop_loss_price: float) -> Dict[str, Any]:
        """
        Обновление стоп-лосса позиции (трейлинг)
        
        Args:
            symbol: Торговая пара
            direction: Направление позиции ("long" или "short")
            stop_loss_price: Новая цена стоп-лосса
        """
        try:
            position_type = 1 if direction.lower() == "long" else 2
            
            params = f"symbol={symbol}&stop_loss_price={stop_loss_price}&position_type={position_type}"
            
            logger.info(f"Обновление стоп-лосса: {direction} {symbol}, новый SL: {stop_loss_price}")
            
            return self._make_request("/api/v1/private/position/set-stop-loss", params)
            
        except Exception as e:
            logger.error(f"Ошибка обновления стоп-лосса: {e}")
            return {"error": f"Ошибка обновления стоп-лосса: {str(e)}"}

    def get_positions(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Получение открытых позиций"""
        try:
            params = f"symbol={symbol}" if symbol else ""
            return self._make_request("/api/v1/private/position/open_positions", params)
        except Exception as e:
            logger.error(f"Ошибка получения позиций: {e}")
            return {"error": f"Ошибка получения позиций: {str(e)}"}

    def get_account_info(self) -> Dict[str, Any]:
        """Получение информации об аккаунте"""
        try:
            return self._make_request("/api/v1/private/account/assets", "")
        except Exception as e:
            logger.error(f"Ошибка получения информации об аккаунте: {e}")
            return {"error": f"Ошибка получения информации об аккаунте: {str(e)}"}
