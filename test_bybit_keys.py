import os
import time
import hmac
import hashlib
import requests

# === 1️⃣ Считываем ключи из ENV ===
API_KEY    = os.getenv("BYBIT_API_KEY", "").strip()
API_SECRET = os.getenv("BYBIT_API_SECRET", "").strip()
BASE_URL   = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com").strip()

if not API_KEY or not API_SECRET:
    print("❌ BYBIT_API_KEY или BYBIT_API_SECRET не заданы в переменных окружения")
    exit(1)

print(f"🔑 KEY: {API_KEY[:5]}...  ✅ Секрет: {'*'*8}")
print(f"🌐 URL: {BASE_URL}")

# === 2️⃣ Подготовка параметров ===
timestamp = str(int(time.time() * 1000))
params = {
    "api_key": API_KEY,
    "timestamp": timestamp,
    "recv_window": "5000"
}

# === 3️⃣ Сортируем параметры и создаём строку ===
sorted_params = sorted(params.items())
param_str = "&".join([f"{k}={v}" for k, v in sorted_params])

# === 4️⃣ Подписываем ===
signature = hmac.new(
    bytes(API_SECRET, "utf-8"),
    bytes(param_str, "utf-8"),
    hashlib.sha256
).hexdigest()

# === 5️⃣ Отправляем тестовый запрос ===
headers = {
    "X-BAPI-API-KEY": API_KEY,
    "X-BAPI-SIGN": signature,
    "X-BAPI-TIMESTAMP": timestamp,
    "X-BAPI-RECV-WINDOW": "5000",
    "Content-Type": "application/json"
}

url = f"{BASE_URL}/v5/account/info"

print(f"🚀 Отправляем GET → {url}")
response = requests.get(url, headers=headers)
data = response.json()

print("📩 Ответ от Bybit:")
print(data)

# === 6️⃣ Проверка результата ===
if data.get("retCode") == 0:
    print("✅ ✅ ✅  Ключи и URL — РАБОТАЮТ!  ✅ ✅ ✅")
else:
    print("❌ Ошибка:", data.get("retCode"), data.get("retMsg"))
    if data.get("retCode") == 10004:
        print("👉 Ошибка подписи. Проверь SECRET/URL/Testnet/Mainnet")
    elif data.get("retCode") == 10007:
        print("👉 Ключ верный, но нет данных аккаунта — это всё равно успех ✅")
    else:
        print("👉 Проверь права ключа или URL")
