import asyncio
import json
import websockets
import time
from dataclasses import dataclass

@dataclass
class WSConfig:
    symbol: str
    market_type: str      # "linear" или "spot"
    testnet: bool
    intervals: tuple
    only_on_confirmed_close: bool = True

class BybitWSKlines:
    def __init__(self, config: WSConfig, callback):
        self.config = config
        self.callback = callback
        self.ws_url = self._get_ws_url()
        self.ping_interval = 20
        self.reconnect_delay = 5
        self.last_pong = time.time()

    def _get_ws_url(self):
        if self.config.market_type == "spot":
            return "wss://stream-testnet.bybit.com/v5/public/spot" if self.config.testnet else "wss://stream.bybit.com/v5/public/spot"
        else:
            return "wss://stream-testnet.bybit.com/v5/public/linear" if self.config.testnet else "wss://stream.bybit.com/v5/public/linear"

    async def run_forever(self):
        while True:
            try:
                async with websockets.connect(self.ws_url, ping_interval=None) as ws:
                    await self._subscribe(ws)
                    async for message in ws:
                        await self._handle_message(message)
            except Exception as e:
                print(f"[WS] Error: {e}, reconnecting in {self.reconnect_delay}s...")
                await asyncio.sleep(self.reconnect_delay)

    async def _subscribe(self, ws):
        for interval in self.config.intervals:
            topic = f"kline.{interval}.{self.config.symbol}"
            sub_msg = {"op": "subscribe", "args": [topic]}
            await ws.send(json.dumps(sub_msg))
            print(f"[WS] Subscribed to {topic}")
        # Пингуем вручную
        asyncio.create_task(self._ping(ws))

    async def _ping(self, ws):
        while True:
            try:
                await ws.send(json.dumps({"op": "ping"}))
                await asyncio.sleep(self.ping_interval)
            except:
                break

    async def _handle_message(self, message):
        data = json.loads(message)
        if "topic" in data and "data" in data:
            topic = data["topic"]
            parts = topic.split(".")
            if len(parts) >= 3 and parts[0] == "kline":
                interval = parts[1]
                symbol = parts[2]
                for candle in data["data"]:
                    start = int(candle["start"])
                    end = int(candle["end"])
                    confirm = candle.get("confirm", False)
                    if self.config.only_on_confirmed_close and not confirm:
                        continue
                    # передаём в колбэк
                    self.callback(symbol, interval, {
                        "start": start,
                        "end": end,
                        "open": float(candle["open"]),
                        "high": float(candle["high"]),
                        "low": float(candle["low"]),
                        "close": float(candle["close"]),
                        "volume": float(candle["volume"]),
                        "turnover": float(candle.get("turnover", 0)),
                        "confirm": confirm
                    })