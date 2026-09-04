"""scoring.py — Motor de scoring deterministico (sin LLM).

Reemplaza LLMJudge con reglas explicitas y pesos configurables.
Entra TokenFeatures (con senales on-chain enriquecidas) y sale Verdict:
accion entrada/TP/SL + razones + risk_flags, todo auditable y reproducible.

Capa 3 del marco SRM: si existe srm_model.pkl (entrenado offline con
srm_trainer.py) lo usa via predict_proba; cualquier fallo de carga o
inferencia degrada a la logica determinista (graceful degradation).
Los hard overrides de seguridad (honeypot/rug/safety) NUNCA se omiten.
"""

import logging
import math
import pathlib
import pickle
from dataclasses import dataclass, field

# Capa 1 (paper_logger) provee FEATURE_NAMES/vectorize: unica fuente de verdad
# del orden de columnas. Si falla el import, el motor sigue determinista.
try:
    import paper_logger as _plog
except Exception:                                 # noqa: BLE001 - degradacion
    _plog = None

SRM_MODEL_PATH = pathlib.Path(__file__).parent / "srm_model.pkl"


@dataclass
class Verdict:
    action: str = "SKIP"          # ENTER / WATCH / SKIP
    confidence: float = 0.0       # 0..1
    entry_price: float = 0.0
    take_profit: list = field(default_factory=list)   # [(+60%, 40%), (+150%, 30%)]
    stop_loss: float = -0.35      # -35%
    reasons: list = field(default_factory=list)
    risk_flags: list = field(default_factory=list)
    dimension_scores: dict = field(default_factory=dict)


def _clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


class ScoringEngine:
    """Deterministic multi-dimension scoring. No LLM.

    Dimensiones (pesos en CFG['scoring_weights']):
      - momentum:  5m/1h price change + kline pattern + volatilidad
      - smart_money: confluence + holder P&L ratio + netflow
      - liquidity:  pool depth + volume + slippage proxy
      - safety:     renounce + top10 + bundler + tax
      - dev:        dev_score (si disponible)
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.w = cfg.get("scoring_weights", {})
        self.thresholds = cfg.get("scoring_thresholds", {})
        # --- SRM capa 3: carga segura del modelo (None => determinista) ---
        self.ml_model = None
        self.ml_feature_names = None
        self.ml_meta: dict = {}
        self._load_srm_model()

    def _load_srm_model(self):
        try:
            if not SRM_MODEL_PATH.exists():
                return
            with SRM_MODEL_PATH.open("rb") as fh:
                bundle = pickle.load(fh)
            mdl = bundle.get("model")
            if mdl is None or not hasattr(mdl, "predict_proba"):
                return
            names = bundle.get("feature_names")
            if not names and _plog is not None:
                names = _plog.FEATURE_NAMES
            self.ml_model = mdl
            self.ml_feature_names = list(names or [])
            self.ml_meta = {k: bundle.get(k) for k in
                            ("bound", "empirical_error", "n_samples", "hierarchy_name", "trained_at")}
            logging.info("[SRM] modelo ML cargado (%s, n=%s, cota=%.4f)",
                         self.ml_meta.get("hierarchy_name"), self.ml_meta.get("n_samples"),
                         self.ml_meta.get("bound") or -1.0)
        except Exception as e:                    # noqa: BLE001 - degradacion silenciosa
            self.ml_model = None
            logging.info("[SRM] srm_model.pkl no usable (%s) -> motor determinista", e)

    def _token_feature_dict(self, f) -> dict:
        """TokenFeatures -> dict con las mismas claves que el training set
        (app._feat + rug_ratio). Orden fijo garantizado por vectorize()."""
        return {
            "volatility": f.volatility, "smart_money_netflow": f.smart_money_netflow,
            "holder_pnl_ratio": f.holder_pnl_ratio, "trader_buy_ratio": f.trader_buy_ratio,
            "chg_1h": f.chg_1h, "chg_5m": f.chg_5m, "buy_ratio": f.buy_ratio,
            "turnover": f.turnover, "liquidity": f.liquidity, "mcap": f.mcap,
            "age_min": f.age_min, "bundler": f.bundler, "dev_hold": f.dev_hold,
            "top10": f.top10, "buy_tax": f.buy_tax, "sell_tax": f.sell_tax,
            "rug_ratio": f.rug_ratio, "smart_degen": f.smart_degen,
            "renowned": f.renowned, "sm_confluence": f.sm_confluence,
            "sniper_count": f.sniper_count,
            "dev_score": (float(f.dev_eval) if f.dev_eval is not None else None),
        }

    def _ml_proba(self, f) -> float | None:
        """P(y=ENTER | features) del modelo SRM. None si no hay modelo o fallo."""
        if self.ml_model is None or _plog is None:
            return None
        try:
            vec = _plog.vectorize(self._token_feature_dict(f), self.ml_feature_names)
            proba = float(self.ml_model.predict_proba([vec])[0][1])
            return proba if math.isfinite(proba) else None
        except Exception as e:                    # noqa: BLE001 - degradacion
            logging.warning("[SRM] inferencia fallo (%s) -> degradando a determinista", e)
            self.ml_model = None                  # no reintentar en esta instancia
            return None

    def score(self, f) -> Verdict:
        v = Verdict(entry_price=f.price)
        flags = []
        reasons = []

        # --- Dimension scores (0..1 each) ---
        dim_momentum = self._score_momentum(f, flags, reasons)
        dim_smart = self._score_smart_money(f, flags, reasons)
        dim_liquidity = self._score_liquidity(f, flags, reasons)
        dim_safety = self._score_safety(f, flags, reasons)
        dim_dev = self._score_dev(f, flags, reasons)

        weights = self.w
        total_w = sum(weights.values()) or 1.0
        composite = (
            weights.get("momentum", 30) * dim_momentum +
            weights.get("smart_money", 25) * dim_smart +
            weights.get("liquidity", 15) * dim_liquidity +
            weights.get("safety", 15) * dim_safety +
            weights.get("dev", 15) * dim_dev
        ) / total_w

        v.dimension_scores = dict(
            momentum=round(dim_momentum, 3),
            smart_money=round(dim_smart, 3),
            liquidity=round(dim_liquidity, 3),
            safety=round(dim_safety, 3),
            dev=round(dim_dev, 3),
        )
        v.risk_flags = flags
        v.reasons = reasons
        v.stop_loss = -self.cfg.get("hard_stop_pct", 0.35)
        v.take_profit = self.cfg.get("tp_ladder", [(0.60, 0.40), (1.50, 0.30)])

        # --- SRM capa 3: blend ML con degradacion a determinista ---
        # El score efectivo es predict_proba si hay modelo; si no, composite.
        # dimension_scores deterministas se conservan como auditoria.
        ml_proba = self._ml_proba(f)
        effective = composite
        if ml_proba is not None:
            effective = ml_proba
            v.reasons.append(f"SRM ML proba={ml_proba:.3f}")
            logging.info("[SRM] %s | composite=%.3f -> ml_proba=%.3f",
                         f.symbol_safe, composite, ml_proba)

        v.confidence = round(_clamp(effective), 3)

        # --- Logging de dimensiones (auditoria en terminal) ---
        logging.info(
            "[SCORING] %s | dim: momentum=%.3f smart_money=%.3f liquidity=%.3f safety=%.3f dev=%.3f | composite=%.3f",
            f.symbol_safe, dim_momentum, dim_smart, dim_liquidity, dim_safety, dim_dev, composite)

        # --- Decision (mismo umbral; dim_safety sigue siendo obligatoria) ---
        enter_th = self.thresholds.get("enter", 0.75)
        watch_th = self.thresholds.get("watch", 0.40)

        if effective >= enter_th and dim_safety >= 0.3:
            v.action = "ENTER"
        elif effective >= watch_th:
            v.action = "WATCH"
        else:
            v.action = "SKIP"

        logging.info(
            "[VERDICT] %s | action=%s confidence=%.3f flags=%s reasons=%s",
            f.symbol_safe, v.action, v.confidence, flags, reasons[:3])

        # Hard overrides (never enter these)
        if f.honeypot:
            v.action = "SKIP"
            v.confidence = 0.0
            v.reasons.insert(0, "HONEYPOT - no entrar")
        elif f.rug_ratio > 0.5:
            v.action = "SKIP"
            v.reasons.insert(0, f"rug_ratio {f.rug_ratio:.0%} demasiado alto")
        elif dim_safety < 0.2:
            v.action = "SKIP"
            v.reasons.insert(0, "seguridad inaceptable")

        return v

    # ---------------------------------------------------- dimensiones
    def _score_momentum(self, f, flags, reasons):
        s5 = _clamp((f.chg_5m + 0.05) / 0.30)       # -5%→0, +25%→1
        s1 = _clamp((f.chg_1h + 0.10) / 0.60)       # -10%→0, +50%→1

        # kline pattern boost/penalty
        pattern_score = 0.5
        if f.kline_pattern == "uptrend":
            pattern_score = 1.0
            reasons.append("patron de velas: uptrend confirmado")
        elif f.kline_pattern == "breakdown":
            pattern_score = 0.1
            flags.append("bearish_pattern")
            reasons.append("patron de velas: breakdown")
        elif f.kline_pattern == "distribution":
            pattern_score = 0.2
            flags.append("bearish_pattern")
            reasons.append("patron de velas: distribucion en altos")
        elif f.kline_pattern == "basing":
            pattern_score = 0.5
        elif f.kline_pattern == "chop":
            pattern_score = 0.4

        # volatility: moderate is good, extreme is risky
        vol_score = 1.0 - _clamp(abs(f.volatility) / 10.0)  # vol>10% → 0
        if f.volatility > 5:
            flags.append("high_volatility")

        score = 0.45 * s5 + 0.30 * s1 + 0.15 * pattern_score + 0.10 * vol_score

        if f.chg_1h <= self.cfg.get("momentum_reject_chg1h", -0.12):
            score *= 0.4
            flags.append("fading_1h")
            reasons.append("1h en caida, momentum degradado")
        return _clamp(score)

    def _score_smart_money(self, f, flags, reasons):
        # confluence (existing)
        s_conf = _clamp(math.log10(1 + f.sm_confluence) / 2.5)

        # holder P&L ratio (new)
        s_holder_pnl = _clamp(f.holder_pnl_ratio)
        if f.holder_pnl_ratio > 0.5:
            reasons.append(f"holders en ganancia: {f.holder_pnl_ratio:.0%}")
        elif f.holder_pnl_ratio > 0 and f.holder_pnl_ratio < 0.3:
            flags.append("holders_in_loss")

        # smart money netflow (new)
        s_netflow = 0.5
        if f.smart_money_netflow > 0:
            s_netflow = _clamp(0.5 + f.smart_money_netflow / 10000)
            reasons.append(f"smart money acumulando: ${f.smart_money_netflow:,.0f}")
            flags.append("smart_money_present")
        elif f.smart_money_netflow < 0:
            s_netflow = _clamp(0.5 + f.smart_money_netflow / 10000)
            flags.append("smart_money_distributing")

        # trader buy ratio (new)
        s_trader = _clamp(f.trader_buy_ratio)

        score = 0.35 * s_conf + 0.25 * s_holder_pnl + 0.20 * s_netflow + 0.20 * s_trader
        return _clamp(score)

    def _score_liquidity(self, f, flags, reasons):
        liq = f.liquidity
        # liquidity depth
        s_liq = _clamp(math.log10(max(1, liq)) / 4.0)  # $10K→0.75, $100K→1.0
        if liq < 15000:
            flags.append("thin_liquidity")
        if liq > 50000:
            reasons.append(f"liquidez saludable: ${liq:,.0f}")

        # volume vs mcap (turnover proxy)
        s_vol = _clamp((f.vol_1h / max(1, f.mcap)) / 2.0) if f.mcap > 0 else 0

        return _clamp(0.6 * s_liq + 0.4 * s_vol)

    def _score_safety(self, f, flags, reasons):
        s = 0.0
        if f.renounced_mint and f.renounced_freeze:
            s += 0.5
            reasons.append("mint + freeze renunciados")
        elif f.renounced_mint:
            s += 0.3
            reasons.append("mint renunciado")
        else:
            flags.append("mint_not_renounced")

        s += 0.5 * _clamp((0.40 - f.top10) / 0.40)
        if f.top10 > 0.3:
            flags.append("top10_concentrated")

        if f.bundler > 0.3:
            flags.append("bot_heavy")
            s -= 0.15
        if f.buy_tax > 0.05 or f.sell_tax > 0.05:
            flags.append("high_tax")
            s -= 0.1
        if f.rug_ratio > 0.2:
            flags.append("rug_elevated")
            s -= 0.1
        return _clamp(s)

    def _score_dev(self, f, flags, reasons):
        if f.dev_eval is not None:
            score = _clamp(f.dev_eval)
            if score < 0.3:
                flags.append("dev_low_reputation")
                reasons.append(f"dev reputation bajo: {score:.0%}")
            elif score > 0.7:
                reasons.append(f"dev solido: {score:.0%}")
            return score
        return 0.5  # neutral si no hay datos
