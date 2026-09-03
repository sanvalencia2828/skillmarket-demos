"""gmgn_client.py — Capa unificada de acceso a gmgn-cli.

Elimina la duplicacion entre aitrader/app.py y sistema_volatilidad/gmgn_bridge.py:
un solo throttler (LeakyBucket con pesos reales), un solo cache TTL,
un solo resolver Windows-safe, un solo manejo de 449.

Uso:
    from gmgn_client import gmgn, GMGNClient, RateLimitError
    data = gmgn.cli("info", ["token", "info", "--chain", "sol", "--address", "..."])
"""

import json
import logging
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    encoding="utf-8",
)

GMGN_BAN_WAIT = float(os.environ.get("GMGN_BAN_WAIT", "210"))

_GMGN_PREFIX = None


def _warmup_dns():
    """Resuelve openapi.gmgn.ai en Python para calentar el cache de DNS de Windows.
    Node.js (c-ares) tiene un bug intermitente donde no resuelve el hostname;
    resolver primero desde Python (que usa el resolver de Windows) lo arregla.
    Espera hasta que el DNS se recupere (max 120s) porque el DNS es intermitente."""
    import socket
    for i in range(60):
        try:
            socket.gethostbyname("openapi.gmgn.ai")
            if i > 0:
                logging.info("[DNS] resuelto despues de %s intentos", i + 1)
            return
        except Exception:
            time.sleep(2)


def _gmgn_prefix():
    """Resuelve gmgn-cli a un argv que CreateProcess puede ejecutar (.cmd no es .exe)."""
    global _GMGN_PREFIX
    if _GMGN_PREFIX is not None:
        return _GMGN_PREFIX
    if os.name == "nt":
        npm = pathlib.Path(os.environ.get("APPDATA", "")) / "npm"
        js = npm / "node_modules" / "gmgn-cli" / "dist" / "index.js"
        node = shutil.which("node")
        if node and js.exists():
            _GMGN_PREFIX = [node, str(js)]
            return _GMGN_PREFIX
    found = shutil.which("gmgn-cli")
    _GMGN_PREFIX = [found] if found else ["gmgn-cli"]
    return _GMGN_PREFIX


class RateLimitError(RuntimeError):
    """429 RATE_LIMIT_BANNED/EXCEEDED de GMGN."""
    pass


class LeakyBucket:
    """Capacidad 20, recarga 10 peso/s (mitad del limite real => margen)."""

    CAPACITY = 20.0
    REFILL = 10.0

    def __init__(self):
        self.tokens = self.CAPACITY
        self.ts = time.monotonic()
        self.lock = threading.Lock()
        self.last_call = 0.0

    def acquire(self, weight: float, min_gap: float = 0.35):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.CAPACITY, self.tokens + (now - self.ts) * self.REFILL)
                self.ts = now
                gap_ok = (now - self.last_call) >= min_gap
                if self.tokens >= weight and gap_ok:
                    self.tokens -= weight
                    self.last_call = now
                    return
                need = max(weight - self.tokens, 0.0)
                wait = max(need / self.REFILL, (min_gap - (now - self.last_call)) if not gap_ok else 0.0)
            time.sleep(min(wait + 0.05, 2.0))


WEIGHTS = {
    "info": 1, "security": 1, "pool": 1, "trending": 1, "search": 1,
    "kline": 2, "hot-searches": 3, "trenches": 3, "signal": 3,
    "holders": 5, "traders": 5,
}
CACHE_TTL = {
    "info": 60, "security": 60, "pool": 60,
    "holders": 120, "traders": 120,
    "hot-searches": 45, "trenches": 60, "trending": 30,
    "kline": 60, "signal": 30,
}

_429_RE = re.compile(r"429|RATE_LIMIT_(BANNED|EXCEEDED)", re.I)


class GMGNClient:
    """Cliente unificado gmgn-cli: throttle + cache + 429 fail-fast."""

    def __init__(self):
        self.bucket = LeakyBucket()
        self._cache = {}

    def cli(self, subcommand: str, args: list, timeout: int = 90, env: dict | None = None):
        key = (subcommand, tuple(args))
        ttl = CACHE_TTL.get(subcommand)
        if ttl:
            hit = self._cache.get(key)
            if hit and time.monotonic() - hit[0] < ttl:
                logging.info("cache hit: %s %s", subcommand, " ".join(args[:4]))
                return hit[1]

        cmd = _gmgn_prefix() + list(args) + ["--raw"]
        self.bucket.acquire(WEIGHTS.get(subcommand, 3))

        max_attempts = 2 if subcommand in ("trending", "trenches") else 3
        for attempt in range(max_attempts):
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout, env=env)
            except FileNotFoundError:
                logging.error("gmgn-cli/node no encontrado en PATH.")
                return None
            except subprocess.TimeoutExpired:
                logging.warning("CLI timeout %ss: %s", timeout, subcommand)
                if attempt < max_attempts - 1:
                    time.sleep(5)
                    continue
                return None

            out = (r.stdout or "").strip()
            err = (r.stderr or "").strip()
            blob = f"{err} {out}"

            if _429_RE.search(blob):
                m = re.search(r"resets at ([0-9\-: ]+GMT[^\)]*)", blob)
                reset = m.group(1) if m else f"~{GMGN_BAN_WAIT:.0f}s"
                logging.error("429 RATE LIMIT. Sin reintento (extiende el ban). Reset: %s", reset)
                raise RateLimitError(reset)

            # Errores transitorios de red/DNS — reintentar con backoff progresivo
            if any(k in blob.lower() for k in ("enoent", "getaddrinfo", "connection", "econnreset", "timeout")):
                if attempt < max_attempts - 1:
                    wait = min(3 * (attempt + 1), 6)
                    logging.warning("CLI error transitorio (%s), retry %s/%s (wait %ss): %s",
                                    subcommand, attempt + 1, max_attempts, wait, err[:80])
                    time.sleep(wait)
                    continue

            if r.returncode == 0 and out:
                try:
                    data = json.loads(out)
                except json.JSONDecodeError:
                    logging.error("Salida no-JSON: %s", out[:200])
                    return None
                if ttl:
                    self._cache[key] = (time.monotonic(), data)
                return data

            logging.error("CLI fallo (%s): %s", subcommand, (err or f"exit {r.returncode}")[:300])
            return None

    def clear_cache(self):
        self._cache.clear()


gmgn = GMGNClient()
