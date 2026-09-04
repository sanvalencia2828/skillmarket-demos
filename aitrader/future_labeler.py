"""future_labeler.py — Opción A: etiquetado diferido en tiempo real (SRM).

Script de BACKGROUND (python future_labeler.py), nunca corre dentro del server.
Cada ciclo (60s):
  1. Lee outputs/paper_trades.jsonl
  2. Busca registros sin target_return cuyo timestamp tenga > 300s (5 min)
  3. Pide el precio actual del token via gmgn-cli (token info) usando el
     cliente unificado (throttle + cache + deteccion 429 de gmgn_client.py)
  4. Calcula  return_pct = (future_price - entry_price) / entry_price
  5. Escribe en el MISMO registro:  future_price, target_return,
     target_class (=1 si return > 0.01, si no 0), labeled_at
  6. sleep(60) y repite

Seguridad:
- 429 de GMGN  -> sleep(180) y continua (el script no cae).
- Token que falla -> descarte silencioso de esta pasada (se reintenta hasta
  MAX_ATTEMPTS veces para no quemar cuota con tokens delistados).
- Escritura atomica: re-lee el archivo fresco justo antes de os.replace para
  minimizar la ventana de carrera con los appends del servidor; las lineas
  nuevas del servidor se conservan verbatim.
- Los registros EVM sin campo chain se omiten (cadena indeterminable);
  paper_logger ahora registra chain para los nuevos.
"""

import datetime
import json
import logging
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

from gmgn_client import gmgn, RateLimitError           # throttle + cache + 429

JSONL_PATH = ROOT / "outputs" / "paper_trades.jsonl"
MIN_AGE_S = 300             # > 5 minutos
LABEL_TH = 0.01             # target_class = 1 si retorno > 1%
CYCLE_SLEEP_S = 60
RATE_LIMIT_SLEEP_S = 180
MAX_ATTEMPTS = 3            # descartes tolerados por registro

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s", encoding="utf-8")
log = logging.getLogger("labeler")

_fail_count: dict = {}      # (address, ts) -> intentos fallidos


def _epoch(rec: dict) -> float:
    raw = rec.get("ts") or rec.get("timestamp")
    if not raw:
        return 0.0
    try:
        return datetime.datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return 0.0


def _entry_price(rec: dict) -> float:
    v = rec.get("verdict") or {}
    try:
        p = v.get("entry_price")
        if p is None:
            p = rec.get("entry_price")
        return float(p or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _infer_chain(rec: dict) -> str | None:
    ch = rec.get("chain")
    if ch:
        return str(ch)
    addr = str(rec.get("address") or "")
    # base58 (Solana) se deduce; 0x... es ambiguo (bsc/base/eth/robinhood)
    return "sol" if (addr and not addr.startswith("0x")) else None


def fetch_current_price(address: str, chain: str) -> float | None:
    """Precio actual via gmgn-cli token info (normaliza price anidado/escalar)."""
    data = gmgn.cli("info", ["token", "info", "--chain", chain, "--address", address], timeout=45)
    if not isinstance(data, dict):
        return None
    info = data.get("data") if isinstance(data.get("data"), dict) else data
    p = info.get("price")
    if isinstance(p, dict):
        p = p.get("price")
    try:
        val = float(p)
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def label_pass() -> int:
    """Una pasada de etiquetado. Devuelve cuantos registros etiqueto.
    Propaga RateLimitError hacia el bucle principal (sleep 180)."""
    if not JSONL_PATH.exists():
        return 0

    raw_lines = JSONL_PATH.read_text(encoding="utf-8").splitlines()
    records = []
    for ln in raw_lines:
        ln = ln.strip()
        if not ln:
            continue
        try:
            records.append(json.loads(ln))
        except json.JSONDecodeError:
            records.append(ln)                 # linea corrupta se preserva

    now = time.time()
    pending = []
    for rec in records:
        if not isinstance(rec, dict) or not isinstance(rec.get("features"), dict):
            continue
        if rec.get("target_return") is not None:
            continue                            # ya etiquetado
        ts = _epoch(rec)
        if ts <= 0 or (now - ts) <= MIN_AGE_S:
            continue                            # demasiado fresco
        entry = _entry_price(rec)
        addr = str(rec.get("address") or "")
        chain = _infer_chain(rec)
        if entry <= 0 or not addr or chain is None:
            continue
        key = (addr, rec.get("ts") or rec.get("timestamp"))
        if _fail_count.get(key, 0) >= MAX_ATTEMPTS:
            continue                            # descartado permanentemente
        pending.append((rec, entry, chain, addr, key))

    if not pending:
        return 0

    labeled: dict = {}
    for rec, entry, chain, addr, key in pending:
        fp = None
        try:
            fp = fetch_current_price(addr, chain)
        except RateLimitError:
            raise                               # bucle principal: sleep(180)
        except Exception as e:
            log.debug("token %s fallo (descartado esta pasada): %s", addr[:10], e)
        if fp:
            ret = (fp - entry) / entry
            rec["future_price"] = fp
            rec["target_return"] = round(ret, 6)
            rec["target_class"] = 1 if ret > LABEL_TH else 0
            rec["labeled_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
            labeled[key] = rec
        else:
            _fail_count[key] = _fail_count.get(key, 0) + 1

    if not labeled:
        return 0

    # Re-lectura fresca + replace atomico (los appends del servidor entre
    # lecturas se conservan verbatim; ventana residual ~ms)
    fresh = JSONL_PATH.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in fresh:
        updated = None
        try:
            r = json.loads(ln)
            if isinstance(r, dict):
                k = (str(r.get("address") or ""), r.get("ts") or r.get("timestamp"))
                updated = labeled.get(k)
        except json.JSONDecodeError:
            pass
        out.append(json.dumps(updated, ensure_ascii=False) if updated is not None else ln)
    tmp = JSONL_PATH.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(out) + "\n", encoding="utf-8")
    os.replace(tmp, JSONL_PATH)
    return len(labeled)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    log.info("[LABELER] etiquetado diferido: ciclo=%ss, edad minima=%ss, umbral=%s",
             CYCLE_SLEEP_S, MIN_AGE_S, LABEL_TH)
    while True:
        try:
            n = label_pass()
            if n:
                log.info("[LABELER] %s registros etiquetados", n)
            time.sleep(CYCLE_SLEEP_S)
        except RateLimitError as e:
            log.warning("[LABELER] GMGN 429 (%s) -> sleep %ss", e, RATE_LIMIT_SLEEP_S)
            time.sleep(RATE_LIMIT_SLEEP_S)
        except Exception as e:
            log.warning("[LABELER] error no fatal (%s) -> continua", e)
            time.sleep(CYCLE_SLEEP_S)


if __name__ == "__main__":
    sys.exit(main())
