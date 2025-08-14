# trail_engine.py - Продвинутый движок трейлинга 1:1 с Pine Script
import os, time, threading, math, logging, requests, json
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple

log = logging.getLogger("trail_engine")

# ---------- helpers ----------
def now_ms() -> int: 
    return int(time.time() * 1000)

def normalize_symbol(sym: str) -> str:
    """TV шлет ETHUSDT.P — для Bybit нужен ETHUSDT (linear)"""
    if sym.endswith(".P"): 
        return sym[:-2]
    return sym

def side_is_long(direction: str) -> bool:
    return direction.lower() in ("long", "buy")

def ceil_to_tick(x: float, tick: float) -> float:
    return round(round(x / tick) * tick, 10)

# ---------- market feeds (Bybit public REST) ----------
BYBIT_PUBLIC = "https://api.bybit.com"

# Кэш для метаданных символов (5 минут TTL)
SYMBOL_CACHE = {}
CACHE_TTL = 300  # 5 минут

def get_tick_size(symbol: str) -> float:
    """Получить tick size с кэшированием"""
    now = time.time()
    cache_key = f"tick_{symbol}"
    
    # Проверяем кэш
    if cache_key in SYMBOL_CACHE:
        data, timestamp = SYMBOL_CACHE[cache_key]
        if now - timestamp < CACHE_TTL:
            return data
    
    try:
        url = f"{BYBIT_PUBLIC}/v5/market/instruments-info"
        r = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=10)
        r.raise_for_status()
        lst = r.json().get("result", {}).get("list", [])
        tick = float(lst[0]["priceFilter"]["tickSize"]) if lst else 0.01
        
        # Сохраняем в кэш
        SYMBOL_CACHE[cache_key] = (tick, now)
        return tick
    except Exception as e:
        log.warning(f"Failed to get tick size for {symbol}: {e}")
        return 0.01

def round_to_tick(price: float, symbol: str) -> float:
    """Округление цены к tick size символа"""
    tick = get_tick_size(symbol)
    return round(price / tick) * tick

def get_last_price(symbol: str) -> float:
    try:
        url = f"{BYBIT_PUBLIC}/v5/market/tickers"
        r = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=10)
        r.raise_for_status()
        lst = r.json().get("result", {}).get("list", [])
        return float(lst[0]["lastPrice"]) if lst else math.nan
    except Exception as e:
        log.warning(f"Failed to get last price for {symbol}: {e}")
        return math.nan

def get_klines_15m(symbol: str, limit: int = 20) -> list:
    try:
        url = f"{BYBIT_PUBLIC}/v5/market/kline"
        params = {"category": "linear", "symbol": symbol, "interval": "15", "limit": str(limit)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        # ответ: [[start, open, high, low, close, volume, turnover], ...] в возр. порядке
        arr = r.json().get("result", {}).get("list", [])
        # к сожалению Bybit возвращает в ОТНОСИТЕЛЬНОМ порядке — нормализуем в возрастающий
        arr = sorted(arr, key=lambda x: int(x[0]))
        # приводим к dict
        out = [{"t": int(a[0]), "o": float(a[1]), "h": float(a[2]), "l": float(a[3]), "c": float(a[4])} for a in arr]
        return out
    except Exception as e:
        log.warning(f"Failed to get klines for {symbol}: {e}")
        return []

# ---------- trading client (uses your bybit_v5_fixed.py) ----------
try:
    from bybit_v5_fixed import get_client  # см. патч ниже
except ImportError:
    get_client = None

class ClientAdapter:
    """Тонкая обертка над твоим клиентом. Тут только то, что нужно для трейла."""
    def __init__(self):
        if not get_client:
            raise RuntimeError("bybit_v5_fixed.get_client() не найден. Обнови файл по патчу ниже.")
        self.cli = get_client()

    def set_stop_loss(self, symbol: str, side: str, stop_loss: float, reduce_only: bool = True):
        """
        Маппинг на /v5/position/trading-stop.
        Для линейных контрактов достаточно указать stopLoss и side.
        """
        if not self.cli:
            raise RuntimeError("Client not initialized")
        try:
            if hasattr(self.cli, 'update_stop_loss'):
                return self.cli.update_stop_loss(symbol, str(stop_loss))
            else:
                # Fallback для старых версий
                return self.cli.make_request("POST", "/v5/position/trading-stop", {
                    "category": "linear",
                    "symbol": symbol,
                    "stopLoss": str(stop_loss),
                    "slTriggerBy": "LastPrice"
                })
        except Exception as e:
            raise RuntimeError(f"Failed to set stop loss: {e}")

# ---------- data models ----------
@dataclass
class StepCfg:
    enabled: bool = True
    step_pct: float = 2.0
    min_gap_sec: int = 15
    base_gap_sec: int = 10          # базовый антиспам по времени
    base_step_ticks: int = 5        # базовый шаг в тиках

@dataclass
class BarTrailCfg:
    enabled: bool = True
    lookback: int = 4
    buffer_ticks: int = 40

@dataclass
class TrailCfg:
    enabled: bool = True
    percent: float = 1.0
    offset: float = 0.4
    use_arm_after_rr: bool = True
    arm_rr: float = 0.5
    bar: BarTrailCfg = field(default_factory=BarTrailCfg)
    step: StepCfg = field(default_factory=StepCfg)

@dataclass
class PositionState:
    symbol: str
    side: str
    qty: float
    entry: float
    sl_init: float
    tp_init: float
    tick: float
    cfg: TrailCfg
    armed: bool = False
    last_trail: Optional[float] = None
    last_send_ts: int = 0  # ms
    created_ms: int = field(default_factory=now_ms)

    def key(self) -> Tuple[str, str]:
        return (self.symbol, self.side.lower())

# ---------- engine ----------
class TrailEngine:
    def __init__(self, poll_sec: float = 1.0):
        self.states: Dict[Tuple[str, str], PositionState] = {}
        self.poll_sec = poll_sec
        self._stop = False
        self._th: Optional[threading.Thread] = None
        self.debouncer: Dict[str, float] = {}  # для защиты от дублей
        try:
            self.client = ClientAdapter()
        except Exception as e:
            log.warning(f"Failed to initialize client adapter: {e}")
            self.client = None
    
    def _is_duplicate_signal(self, key: str, window_sec: int = 10) -> bool:
        """Проверка на дубликат сигнала в окне времени"""
        now = time.time()
        if key in self.debouncer:
            if now - self.debouncer[key] < window_sec:
                return True
        self.debouncer[key] = now
        # Очистка старых записей
        self.debouncer = {k: v for k, v in self.debouncer.items() if now - v < window_sec * 2}
        return False

    # --- ingestion from TV ---
    def start_from_entry(self, payload: Dict[str, Any]):
        symbol = normalize_symbol(payload["symbol"])
        side = payload["direction"]
        qty = float(payload.get("qty", 0))
        entry = float(payload.get("ref_price") or payload.get("entry_price") or payload.get("price") or 0)
        sl = float(payload.get("stop_loss", 0))
        tp = float(payload.get("take_profit", 0))

        cfg = self._cfg_from_payload(payload.get("trail", {}))
        tick = get_tick_size(symbol)
        ps = PositionState(symbol, side, qty, entry, sl, tp, tick, cfg)

        # ARM сразу, если выключен use_arm_after_rr
        if not cfg.use_arm_after_rr:
            ps.armed = True

        self.states[ps.key()] = ps
        log.info(f"[Trail] init state {ps.key()} | entry={entry} sl={sl} tp={tp} tick={tick}")

    def on_external_trail(self, payload: Dict[str, Any]) -> bool:
        """Вернет True, если апдейт принят (лучше/новее), иначе False."""
        symbol = normalize_symbol(payload["symbol"])
        side = payload["direction"]
        new_tr_val = payload.get("new_trail_stop") or payload.get("new_stop_loss") or payload.get("new_stop")
        
        if new_tr_val is None:
            return False
            
        new_tr = float(new_tr_val)
        key = (symbol, side.lower())
        st = self.states.get(key)
        if not st:
            return False

        # Проверка на дубликат
        dedup_key = f"trail:{symbol}:{side}:{new_tr}"
        if self._is_duplicate_signal(dedup_key, 10):
            return False

        new_tr = ceil_to_tick(new_tr, st.tick)
        if self._is_better(st, new_tr):
            self._apply_trail(st, new_tr, reason="tv_trail")
            return True
        return False

    def on_exit(self, payload: Dict[str, Any]):
        symbol = normalize_symbol(payload["symbol"])
        side = payload["direction"]
        self.states.pop((symbol, side.lower()), None)
        log.info(f"[Trail] cleared state for {(symbol, side)} (exit)")

    # --- background loop ---
    def start(self):
        if self._th and self._th.is_alive(): 
            return
        self._stop = False
        self._th = threading.Thread(target=self._loop, daemon=True)
        self._th.start()
        log.info("[Trail] background loop started")

    def stop(self):
        self._stop = True
        if self._th: 
            self._th.join(timeout=5)

    def _loop(self):
        # простая периодическая модель: опрашиваем раз в poll_sec
        while not self._stop:
            try:
                self._tick()
            except Exception as e:
                log.exception(f"[Trail] tick error: {e}")
            time.sleep(self.poll_sec)

    # --- core tick ---
    def _tick(self):
        if not self.states or not self.client: 
            return
        # группируем запросы — тут по одному
        for key, st in list(self.states.items()):
            price = get_last_price(st.symbol)
            if math.isnan(price): 
                continue

            # ARM check
            if st.cfg.use_arm_after_rr and not st.armed:
                if side_is_long(st.side):
                    moved = price - st.entry
                    need = (st.entry - st.sl_init) * st.cfg.arm_rr
                    st.armed = moved >= need
                else:
                    moved = st.entry - price
                    need = (st.sl_init - st.entry) * st.cfg.arm_rr
                    st.armed = moved >= need

            # compute trail candidate
            cand = self._compute_trail_candidate(st, price)
            if cand is None:
                continue
            cand = ceil_to_tick(cand, st.tick)

            if self._should_send(st, cand):
                self._apply_trail(st, cand, reason="model")

    # --- compute logic one-to-one with Pine ---
    def _compute_trail_candidate(self, st: PositionState, last_price: float) -> Optional[float]:
        if not st.cfg.enabled:
            return None

        # если бар-трейл и вооружен — приоритет бар-трейла
        if st.cfg.bar.enabled and st.armed:
            candles = get_klines_15m(st.symbol, limit=max(10, st.cfg.bar.lookback + 2))
            if len(candles) < st.cfg.bar.lookback + 1:
                return None
            # берем lookback лоу/хи ИЗ ПРЕД. БАРОВ (как [1] в Pine)
            prev = candles[:-1][-st.cfg.bar.lookback:]
            if side_is_long(st.side):
                lb_low = min(c["l"] for c in prev)
                buff = st.cfg.bar.buffer_ticks * st.tick
                base = lb_low - buff
                base = max(base, st.sl_init)
                return base
            else:
                lb_high = max(c["h"] for c in prev)
                buff = st.cfg.bar.buffer_ticks * st.tick
                base = lb_high + buff
                base = min(base, st.sl_init)
                return base

        # --- offset gate: не включаем %-трейл, пока цена не прошла offset от входа
        offset_ok = True
        if st.cfg.offset and st.cfg.offset > 0:
            delta = st.entry * (st.cfg.offset / 100.0)
            if side_is_long(st.side):
                offset_ok = last_price >= (st.entry + delta)
            else:
                offset_ok = last_price <= (st.entry - delta)
        if not offset_ok:
            return None

        # иначе fallback на линейный % трейл от entry
        if side_is_long(st.side):
            pts = st.entry * (st.cfg.percent / 100.0)
            trail = last_price - pts
            return trail
        else:
            pts = st.entry * (st.cfg.percent / 100.0)
            trail = last_price + pts
            return trail

    # --- gating rules ---
    def _should_send(self, st: PositionState, new_trail: float) -> bool:
        # направление и «лучше»
        if not self._is_better(st, new_trail):
            return False

        # --- базовый антиспам по времени и шагу в тиках (как в Pine)
        if st.cfg.step.base_gap_sec > 0:
            if (now_ms() - st.last_send_ts) < st.cfg.step.base_gap_sec * 1000:
                return False

        if st.cfg.step.base_step_ticks > 0 and st.last_trail is not None:
            min_step = st.cfg.step.base_step_ticks * st.tick
            if abs(new_trail - st.last_trail) < min_step:
                return False

        # антиспам по времени (дополнительный)
        if st.cfg.step.enabled:
            if (now_ms() - st.last_send_ts) < st.cfg.step.min_gap_sec * 1000:
                return False
            # шаг в %
            if st.last_trail is not None:
                if side_is_long(st.side):
                    need = st.last_trail * (1.0 + st.cfg.step.step_pct / 100.0)
                    if new_trail < need: 
                        return False
                else:
                    need = st.last_trail * (1.0 - st.cfg.step.step_pct / 100.0)
                    if new_trail > need:
                        return False
        return True

    def _is_better(self, st: PositionState, val: float) -> bool:
        if st.last_trail is None: 
            return True
        if side_is_long(st.side):
            return val > st.last_trail
        else:
            return val < st.last_trail

    # --- apply ---
    def _apply_trail(self, st: PositionState, price: float, reason: str):
        try:
            # Округляем к tick size перед отправкой
            rounded_price = round_to_tick(price, st.symbol)
            self.client.set_stop_loss(st.symbol, st.side, rounded_price, reduce_only=True)
            st.last_trail = rounded_price
            st.last_send_ts = now_ms()
            log.info(f"[Trail] {reason}: {st.symbol} {st.side} -> new SL {rounded_price}")
        except Exception as e:
            log.exception(f"[Trail] apply_trail error: {e}")

    # --- cfg from payload or env defaults ---
    def _cfg_from_payload(self, tr: Dict[str, Any]) -> TrailCfg:
        def get(name, default):
            return tr.get(name, float(os.getenv(f"TRAIL_{name.upper()}", default)))
        
        bar = tr.get("bar_trail", {})
        step = tr.get("step", {})
        
        return TrailCfg(
            enabled=bool(tr.get("enabled", os.getenv("TRAIL_ENABLED", "true").lower() == "true")),
            percent=float(tr.get("percent", os.getenv("TRAIL_PERCENT", 1.0))),
            offset=float(tr.get("offset", os.getenv("TRAIL_OFFSET", 0.4))),
            use_arm_after_rr=bool(tr.get("use_arm_after_rr", os.getenv("TRAIL_USE_ARM_AFTER_RR", "true").lower() == "true")),
            arm_rr=float(tr.get("arm_rr", os.getenv("TRAIL_ARM_RR", 0.5))),
            bar=BarTrailCfg(
                enabled=bool(bar.get("enabled", os.getenv("TRAIL_BAR_ENABLED", "true").lower() == "true")),
                lookback=int(bar.get("lookback", os.getenv("TRAIL_BAR_LOOKBACK", 4))),
                buffer_ticks=int(bar.get("buffer_ticks", os.getenv("TRAIL_BAR_BUFFER_TICKS", 40))),
            ),
            step=StepCfg(
                enabled=bool(step.get("enabled", os.getenv("TRAIL_STEP_ENABLED", "true").lower() == "true")),
                step_pct=float(step.get("step_pct", os.getenv("TRAIL_STEP_PCT", 2.0))),
                min_gap_sec=int(step.get("min_gap_sec", os.getenv("TRAIL_MIN_GAP_SEC", 15))),
                base_gap_sec=int(step.get("base_gap_sec", os.getenv("TRAIL_BASE_GAP_SEC", 10))),
                base_step_ticks=int(step.get("base_step_ticks", os.getenv("TRAIL_BASE_STEP_TICKS", 5))),
            )
        )