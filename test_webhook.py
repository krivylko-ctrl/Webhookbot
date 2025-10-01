import requests
import json

webhook_url = "http://localhost:5000/webhook"

test_open_signal = {
    "event": "open_block",
    "exchange": "bybit",
    "category": "linear",
    "symbol": "ETHUSDT",
    "tv_symbol": "ETHUSDT.P",
    "side": "Buy",
    "legs": [
        {
            "id": "05",
            "orderLinkId": "Buy_12345_1730000000_05",
            "price": "2000.50",
            "qty": "0.01",
            "lev": "25",
            "tp": "2100.00",
            "sl": "1950.00"
        }
    ],
    "oid_prefix": "Buy_12345_1730000000",
    "armBar": 12345,
    "bar_index": 100,
    "time": 1730000000
}

test_cancel_signal = {
    "event": "cancel_block",
    "exchange": "bybit",
    "category": "linear",
    "symbol": "ETHUSDT",
    "tv_symbol": "ETHUSDT.P",
    "oid_prefix": "Buy_12345_1730000000"
}

def test_health():
    print("Testing health endpoint...")
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")

def test_open_block():
    print("Testing open_block signal...")
    print(f"Sending: {json.dumps(test_open_signal, indent=2)}\n")
    
    response = requests.post(
        webhook_url,
        json=test_open_signal,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_cancel_block():
    print("Testing cancel_block signal...")
    print(f"Sending: {json.dumps(test_cancel_signal, indent=2)}\n")
    
    response = requests.post(
        webhook_url,
        json=test_cancel_signal,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == '__main__':
    print("=== TradingView Webhook Test ===\n")
    
    try:
        test_health()
        print("NOTE: Remaining tests will attempt to place real orders on Bybit!")
        print("Make sure you're using testnet or are ready to place real orders.\n")
        
        choice = input("Continue with order tests? (y/n): ")
        if choice.lower() == 'y':
            test_open_block()
    
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Make sure Flask server is running on port 5000.")
    except Exception as e:
        print(f"ERROR: {str(e)}")
