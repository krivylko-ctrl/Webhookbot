"""
Простой тест API подключения к Bybit
"""
import os
import hashlib
import hmac
import time
import requests

def test_api_connection():
    """Тестирует подключение к Bybit API"""
    
    api_key = os.getenv("BYBIT_API_KEY", "")
    api_secret = os.getenv("BYBIT_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("❌ API ключи не найдены в переменных окружения")
        return False
    
    print(f"✓ API ключ найден: {api_key[:8]}...")
    print(f"✓ API секрет найден: {api_secret[:8]}...")
    
    # Тестируем публичный эндпоинт (без подписи)
    print("\n🔄 Тестируем публичный эндпоинт...")
    try:
        response = requests.get("https://api.bybit.com/v5/market/time")
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Публичный API работает: {data}")
        else:
            print(f"❌ Публичный API ошибка: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Ошибка публичного API: {e}")
        return False
    
    # Тестируем приватный эндпоинт (с подписью)
    print("\n🔄 Тестируем приватный эндпоинт...")
    try:
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"
        
        # Создаем подпись для приватного запроса
        param_str = f"accountType=SPOT"
        payload = timestamp + api_key + recv_window + param_str
        signature = hmac.new(
            api_secret.encode(), 
            payload.encode(), 
            hashlib.sha256
        ).hexdigest()
        
        headers = {
            "X-BAPI-API-KEY": api_key,
            "X-BAPI-SIGN": signature,
            "X-BAPI-SIGN-TYPE": "2",
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }
        
        url = "https://api.bybit.com/v5/account/wallet-balance"
        response = requests.get(url, params={"accountType": "SPOT"}, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("retCode") == 0:
                print("✓ Приватный API работает!")
                print(f"✓ Баланс получен: {data}")
                return True
            else:
                print(f"❌ API вернул ошибку: {data}")
                return False
        else:
            print(f"❌ HTTP ошибка: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка приватного API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Тестирование API подключения к Bybit...\n")
    success = test_api_connection()
    
    if success:
        print("\n🎉 API подключение работает! Можно запускать бота.")
    else:
        print("\n⚠️ Проблемы с API подключением.")
        print("\n📋 Возможные причины:")
        print("1. Неверные API ключи")
        print("2. Недостаточные разрешения для API ключа")
        print("3. IP адрес не добавлен в whitelist")
        print("4. API ключ заблокирован или истек")
        print("\n💡 Рекомендации:")
        print("- Проверьте правильность API ключей в настройках Bybit")
        print("- Убедитесь что API ключ имеет разрешения на торговлю и чтение баланса")
        print("- Добавьте IP адрес сервера в whitelist API ключа")