"""smoke_test.py — Prueba en vivo del ScoringEngine con API real.

1. Radar en SOL (trenches completed, filtros estrictos)
2. Toma TOP 2
3. Enriquece con kline + holders + traders (senales reales)
4. ScoringEngine calcula 5 dimensiones y emite Verdict
"""
import logging, sys, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", encoding="utf-8")

# Asegurar que gmgn_client es importable
AITRADER = os.path.join(os.path.expanduser("~"), "gmgn-demos", "aitrader")
if AITRADER not in sys.path:
    sys.path.insert(0, AITRADER)

from app import LiveGMGN, FeatureExtractor, hard_gates, ScoringEngine, CFG, enrich_with_onchain_signals, _f

def main():
    chain = "sol"
    g = LiveGMGN(chain)
    fx = FeatureExtractor(g)
    engine = ScoringEngine(CFG)

    print(f"\n{'='*60}")
    print(f"SMOKE TEST — API REAL | chain={chain} | Mock=OFF")
    print(f"{'='*60}")

    # 1. Radar: migrados con filtros estrictos
    print("\n[1] Ejecutando RADAR (trenches completed, min_mc=50000, max_mc=200000)...")
    args = ["market", "trenches", "--chain", chain, "--type", "completed",
            "--min-marketcap", "50000", "--max-marketcap", "200000",
            "--min-liquidity", "10000",
            "--max-top-holder-rate", "0.2",
            "--max-bundler-rate", "0.2",
            "--max-fresh-wallet-rate", "0.2",
            "--limit", "80"]
    from gmgn_client import gmgn
    data = gmgn.cli("trenches", args)
    if not data:
        print("RADAR sin resultados (rate limit o sin datos). Saliendo.")
        return

    inner = data.get("data", data) if isinstance(data, dict) else data
    cands = inner.get("completed", []) if isinstance(inner, dict) else (inner or [])
    cands = [t for t in cands if _f(t.get("rug_ratio"), 0.0) <= 0.7]
    print(f"   RADAR: {len(cands)} candidatos tras filtro rug_ratio<=0.7")

    if not cands:
        print("Sin candidatos. Saliendo.")
        return

    # 2. Filtrar con hard_gates y tomar TOP 2
    survivors = []
    for t in cands[:30]:
        f = fx.build_from_row(t)
        ok, reason, gate = hard_gates(f)
        if ok:
            survivors.append(f)
        else:
            logging.info("   [GATE %d] %s: %s", gate, f.symbol_safe, reason[:60])

    survivors.sort(key=lambda x: -(x.smart_degen + x.renowned))
    top = survivors[:2]
    print(f"\n[2] TOP 2 candidatos que pasaron hard_gates:")
    for i, f in enumerate(top, 1):
        print(f"   #{i} {f.symbol_safe}  sm={f.sm_confluence}  rug={f.rug_ratio:.2f}  chg1h={f.chg_1h:+.0%}")

    # 3. Enriquecer y Scoring para cada uno
    for i, f in enumerate(top, 1):
        print(f"\n{'─'*60}")
        print(f"[3.{i}] ENRICH + SCORING: {f.symbol_safe}")
        print(f"{'─'*60}")
        enrich_with_onchain_signals(g, f)
        v = engine.score(f)

        print(f"\n   ╔══ VEREDICTO {'═'*40}")
        print(f"   ║ action:     {v.action}")
        print(f"   ║ confidence: {v.confidence:.3f}")
        print(f"   ║ entry:      ${f.price:.6f}")
        print(f"   ║ stop_loss:  {v.stop_loss:.0%}")
        print(f"   ║ take_profit: {v.take_profit}")
        print(f"   ║ dimensions: {v.dimension_scores}")
        print(f"   ║ risk_flags: {v.risk_flags}")
        print(f"   ║ reasons:")
        for r in v.reasons[:5]:
            print(f"   ║   • {r}")
        print(f"   ╚{'═'*50}")

    print(f"\n{'='*60}")
    print("SMOKE TEST COMPLETADO")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
