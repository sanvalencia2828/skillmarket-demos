"""mock_collector.py — Colector de transiciones RL en Modo MOCK (temporal).

En MOCK el scoring solo corre on-demand (/api/enrich); el screener de fondo
(radar rapido) no pasa por engine.score(). Este script acelera la recoleccion
del dataset RL: cada ciclo pregunta al server la lista de tokens mock y los
enriquece uno a uno -> cada enrich genera una transicion rl_transition con la
recompensa calculada por el RL Shadow Mode (reward_engine + position_manager).

Uso:  python mock_collector.py [--target 500] [--interval 20]
Parametros:
  --target N    se detiene al llegar a N transiciones (default 500, el gate)
  --interval S  segundos entre ciclos (default 20)

Solo habla HTTP con el server local: cero cuota GMGN. Robusto: si el server
esta caido duerme y reintenta; Ctrl+C sale limpio.
"""

import argparse
import json
import pathlib
import sys
import time
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

ROOT = pathlib.Path(__file__).parent
sys.stdout.reconfigure(encoding="utf-8")

SERVER = "http://127.0.0.1:8000"
JSONL_PATH = ROOT / "outputs" / "paper_trades.jsonl"
CHAINS = ["sol", "robinhood"]


def _get_json(path: str, timeout: int = 60):
    with urlopen(SERVER + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def count_transitions() -> int:
    if not JSONL_PATH.exists():
        return 0
    n = 0
    for line in JSONL_PATH.read_text(encoding="utf-8").splitlines():
        if '"kind": "rl_transition"' in line or '"kind":"rl_transition"' in line:
            n += 1
    return n


def collect_addresses() -> list:
    """[(address, chain)] de los tokens mock de todas las chains."""
    import urllib.request
    out = []
    for chain in CHAINS:
        try:
            req = urllib.request.Request(
                SERVER + "/api/run",
                data=json.dumps({"chain": chain}).encode("utf-8"),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                d = json.loads(resp.read().decode("utf-8"))
            for dec in d.get("decisions", []):
                dd = dec.get("decision") or dec
                addr = dd.get("address")
                if addr:
                    out.append((addr, chain))
        except (URLError, HTTPError, TimeoutError) as e:
            print(f"[collector] /api/run {chain} fallo: {e}")
    # dedupe manteniendo orden
    seen, uniq = set(), []
    for addr, chain in out:
        if addr not in seen:
            seen.add(addr)
            uniq.append((addr, chain))
    return uniq


def churn_exit(server_addr: str, chain: str, size_sol: float = 0.01):
    """Buy + unmonitor (papel) -> dispara el RL Terminal Bridge -> transicion EXIT.

    Unmonitor y no sell: el unmonitor no alimenta el kill-switch de riesgo,
    asi el churn puede correr indefinidamente."""
    import urllib.request
    req_buy = urllib.request.Request(
        SERVER + "/api/buy",
        data=json.dumps({"address": server_addr, "size_sol": size_sol, "chain": chain}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req_buy, timeout=60) as resp:
        json.loads(resp.read().decode("utf-8"))
    req_un = urllib.request.Request(
        SERVER + "/api/unmonitor",
        data=json.dumps({"address": server_addr}).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req_un, timeout=60) as resp:
        json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Colector RL en modo MOCK")
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--interval", type=int, default=20)
    ap.add_argument("--churn", type=int, default=2,
                    help="tokens buy+unmonitor por ciclo para generar EXIT terminales")
    args = ap.parse_args()

    print(f"[collector] objetivo: {args.target} transiciones | ciclo cada {args.interval}s")
    start_count = count_transitions()
    print(f"[collector] dataset actual: {start_count}")

    while True:
        try:
            n0 = count_transitions()
            if n0 >= args.target:
                print(f"\n[collector] OBJETIVO ALCANZADO: {n0}/{args.target} transiciones. "
                      "Ejecuta: python train_dqn.py (el gate se abrira al reiniciar el server)")
                return 0
            addrs = collect_addresses()
            if not addrs:
                print("[collector] sin tokens mock (server caido?) -> reintento en el proximo ciclo")
                time.sleep(args.interval)
                continue
            ok = err = 0
            for addr, chain in addrs:
                try:
                    _get_json(f"/api/enrich?address={addr}&chain={chain}", timeout=60)
                    ok += 1
                except Exception as e:
                    err += 1
                    print(f"[collector] enrich fallo {addr[:10]}: {e}")
            # churn: buy+unmonitor de N tokens -> EXIT terminales (varianza de reward)
            churned = 0
            if args.churn and addrs:
                import random as _rnd
                for addr, chain in _rnd.sample(addrs, min(args.churn, len(addrs))):
                    try:
                        churn_exit(addr, chain)
                        churned += 1
                    except Exception as e:
                        print(f"[collector] churn fallo {addr[:10]}: {e}")
            n1 = count_transitions()
            eta = (args.target - n1) / max(1, (n1 - n0)) * args.interval if n1 > n0 else float("inf")
            eta_s = f"~{eta / 60:.0f} min" if eta != float("inf") else "n/a"
            print(f"[collector] ciclo: {ok} enriquecidos, {churned} churn (EXIT) | "
                  f"transiciones: {n0} -> {n1}/{args.target} | ETA {eta_s}")
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[collector] detenido por el usuario.")
            return 0
        except Exception as e:
            print(f"[collector] error no fatal ({e}) -> reintento")
            time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
