"""paper_logger.py — Recolector silencioso de datos para entrenamiento SRM.

Capa 1 del marco SRM (aislada): registra cada decision del ScoringEngine
(features + veredicto) en JSONL para entrenar offline con srm_trainer.py.

Reglas:
- Nunca lanza excepciones hacia el llamador: cualquier fallo se traga
  (graceful degradation del sistema determinista).
- Unica fuente de verdad para el vector de features: FEATURE_NAMES +
  vectorize() son usados por srm_trainer.py (entrenamiento) y
  scoring.py (inferencia) para garantizar el mismo orden de columnas.
"""

import dataclasses
import json
import logging
import math
import pathlib
import threading
import time

logging.getLogger("paper_logger").setLevel(logging.INFO)
_log = logging.getLogger("paper_logger")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _log.addHandler(_h)

OUT_DIR = pathlib.Path(__file__).parent / "outputs"
JSONL_PATH = OUT_DIR / "paper_trades.jsonl"

# Orden fijo de columnas para sklearn (entrenamiento == inferencia).
# Correspondencia: claves de app._feat(f) + rug_ratio (anadido en el call site).
FEATURE_NAMES = [
    # senales on-chain (Fase 2)
    "volatility", "smart_money_netflow", "holder_pnl_ratio", "trader_buy_ratio",
    # mercado
    "chg_1h", "chg_5m", "buy_ratio", "turnover", "liquidity", "mcap", "age_min",
    # seguridad
    "bundler", "dev_hold", "top10", "buy_tax", "sell_tax", "rug_ratio",
    # smart money
    "smart_degen", "renowned", "sm_confluence", "sniper_count",
    # dev (None -> 0.5 neutral, igual que _score_dev)
    "dev_score",
]

# Valor neutral por feature cuando falta/None
NEUTRAL = {"dev_score": 0.5}

_LOCK = threading.Lock()


def vectorize(features: dict, feature_names: list | None = None) -> list:
    """dict de features -> lista de floats en orden fijo (sklearn-safe).

    Tolerante: None -> neutral, bool -> 0/1, NaN/inf -> 0.0, strings -> neutral.
    """
    names = feature_names if feature_names is not None else FEATURE_NAMES
    out = []
    for name in names:
        raw = features.get(name)
        if raw is None:
            out.append(float(NEUTRAL.get(name, 0.0)))
        elif isinstance(raw, bool):
            out.append(1.0 if raw else 0.0)
        elif isinstance(raw, (int, float)):
            v = float(raw)
            out.append(0.0 if (math.isnan(v) or math.isinf(v)) else v)
        else:
            out.append(float(NEUTRAL.get(name, 0.0)))
    return out


def _verdict_to_dict(verdict):
    """Acepta Verdict dataclass, dict, u objeto con __dict__."""
    if isinstance(verdict, dict):
        return verdict
    try:
        if dataclasses.is_dataclass(verdict) and not isinstance(verdict, type):
            return dataclasses.asdict(verdict)
    except Exception:
        pass
    try:
        return dict(vars(verdict))
    except Exception:
        pass
    try:
        return dict(verdict)
    except Exception:
        pass
    d = {}
    for k in ("action", "confidence", "entry_price", "take_profit", "stop_loss", "risk_flags"):
        val = getattr(verdict, k, None)
        if val is not None:
            d[k] = val
    return d


def log_decision_for_training(features_dict: dict, verdict, token_address: str,
                              chain: str | None = None) -> dict | None:
    """Registra una decision (features + veredicto) para entrenamiento SRM.

    chain: cadena de la decision (sol/robinhood/...) — el future_labeler la
    necesita para pedir precios de tokens EVM.

    Silencioso por contrato: devuelve el registro escrito o None si fallo;
    JAMAS lanza hacia el llamador (el flujo determinista no se ve afectado).
    """
    try:
        if not isinstance(features_dict, dict):
            features_dict = dict(features_dict or {})
        v = _verdict_to_dict(verdict)
        if not v.get("action"):
            # sin action no hay label para entrenar -> muestra inutil
            _log.warning("paper_logger: verdict sin action (no se registra)")
            return None
        rec = {
            "kind": "training_sample",
            "ts": __import__("datetime").datetime.now(__import__("datetime").timezone.utc)
                     .isoformat(timespec="seconds"),
            "address": token_address,
            "chain": chain,
            "features": {name: features_dict.get(name) for name in FEATURE_NAMES},
            "verdict": {
                "action": v.get("action"),
                "confidence": v.get("confidence"),
                "entry_price": v.get("entry_price"),
                "take_profit": v.get("take_profit"),
                "stop_loss": v.get("stop_loss"),
                "risk_flags": v.get("risk_flags") or [],
            },
        }
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        line = json.dumps(rec, ensure_ascii=False)
        with _LOCK:
            with JSONL_PATH.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        return rec
    except Exception as e:                       # noqa: BLE001 - contrato: silencioso
        _log.warning("paper_logger fallo (no afecta flujo principal): %s", e)
        return None


def log_rl_transition(state, action, reward, next_state=None, done=False):
    """Guarda una transición completa para entrenamiento offline de RL."""
    try:
        record = {
            "kind": "rl_transition",
            "state": state,
            "action": action,
            "reward": float(reward),
            "next_state": next_state,
            "done": 1 if done else 0,
            "timestamp": time.time()
        }
        with _LOCK:
            with JSONL_PATH.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
    except Exception:
        return None  # Degradación silenciosa
