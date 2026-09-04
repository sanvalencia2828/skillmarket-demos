"""srm_trainer.py — Entrenamiento offline SRM (Structural Risk Minimization).

Capa 2 del marco SRM (aislada): script MANUAL, nunca corre con el servidor.

Implementa la jerarquia SRM de Mohri:
  H1: LogisticRegression(penalty='l1')   d_k = 5
  H2: LogisticRegression(penalty='l2')   d_k = 15
  H3: GradientBoostingClassifier(d=2)    d_k = 100

Cota de exceso de error por modelo (proxy de la cota de generalizacion):
  bound_k = Error_Empirico + sqrt((d_k * ln(N)) / N)

Selecciona el modelo con la menor cota total, lo reentrena con TODOS los
datos y lo guarda en srm_model.pkl junto con el orden de features.

Uso:
    python srm_trainer.py                       # lee outputs/paper_trades.jsonl
    python srm_trainer.py --path otro.jsonl     # para tests/datasets alternos
    python srm_trainer.py --out /tmp/model.pkl  # destino alterno del pickle
"""

import argparse
import json
import math
import pathlib
import pickle
import sys
import datetime

import numpy as np

# Solo-dependencias de entrenamiento (sklearn NO se importa desde app.py)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

ROOT = pathlib.Path(__file__).parent
DEFAULT_JSONL = ROOT / "outputs" / "paper_trades.jsonl"
DEFAULT_MODEL = ROOT / "srm_model.pkl"
MIN_SAMPLES = 50

from paper_logger import FEATURE_NAMES, vectorize          # orden unico de columnas

# Jerarquia SRM: (id, descripcion, d_k proxy de dimension VC, fabrica)
HIERARCHY = [
    (1, "LogisticRegression(l1)", 5,
     lambda: LogisticRegression(penalty="l1", solver="liblinear", max_iter=2000)),
    (2, "LogisticRegression(l2)", 15,
     lambda: LogisticRegression(penalty="l2", max_iter=2000)),
    (3, "GradientBoostingClassifier(d2)", 100,
     lambda: GradientBoostingClassifier(max_depth=2, random_state=42)),
]


def load_samples(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Lee JSONL y devuelve (X, y). Solo registros con campo 'features' (kind
    training_sample); el formato legacy de paper trades (solo ENTER) se ignora."""
    X, y = [], []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            feats = rec.get("features")
            if not isinstance(feats, dict):
                continue
            verdict = rec.get("verdict") or {}
            label = 1 if verdict.get("action") == "ENTER" else 0
            X.append(vectorize(feats))
            y.append(label)
    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def empirical_error(model, X: np.ndarray, y: np.ndarray) -> tuple[float, str]:
    """Error empirico por CV estratificada (o train-error si y tiene 1 sola clase)."""
    classes, counts = np.unique(y, return_counts=True)
    if classes.size < 2:
        model.fit(X, y)                          # clase unica: cota degenerada
        return float(1.0 - model.score(X, y)), "train-error (clase unica)"
    n_splits = max(2, min(5, int(counts.min())))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(1.0 - scores.mean()), f"{n_splits}-fold CV"


def srm_bound(err: float, d_k: int, n: int) -> float:
    """Cota proxy de exceso de error: err + sqrt((d_k * ln N) / N)."""
    return err + math.sqrt((d_k * math.log(n)) / n)


def main() -> int:
    ap = argparse.ArgumentParser(description="Entrenamiento offline SRM")
    ap.add_argument("--path", type=pathlib.Path, default=DEFAULT_JSONL, help="JSONL de entrenamiento")
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_MODEL, help="destino del pickle")
    args = ap.parse_args()

    if not args.path.exists():
        print(f"[SRM] sin dataset: {args.path} no existe. Recolecta con paper_logger primero.")
        return 1

    X, y = load_samples(args.path)
    n = len(y)
    if n <= MIN_SAMPLES:
        print(f"[SRM] dataset insuficiente: {n} muestras (se requieren > {MIN_SAMPLES}).")
        return 1

    print(f"[SRM] dataset: {n} muestras | ENTER={int((y == 1).sum())} | no-ENTER={int((y == 0).sum())}")
    print(f"[SRM] features ({len(FEATURE_NAMES)}): {', '.join(FEATURE_NAMES)}\n")

    results = []
    for hid, name, d_k, factory in HIERARCHY:
        model = factory()
        try:
            err, src = empirical_error(model, X, y)
        except Exception as e:                    # p.ej. clase unica con GB
            print(f"  H{hid} {name:32s} FALLO entrenamiento: {e}")
            continue
        bound = srm_bound(err, d_k, n)
        results.append(dict(hid=hid, name=name, d_k=d_k, err=err, bound=bound,
                            factory=factory, src=src))
        print(f"  H{hid} {name:32s} err_emp={err:.4f}  penal(d={d_k})={bound - err:.4f}  "
              f"cota_total={bound:.4f}  [{src}]")

    if not results:
        print("[SRM] ningun modelo entrenable (dataset degenerado).")
        return 1

    winner = min(results, key=lambda r: r["bound"])
    print(f"\n[SRM] ganador: H{winner['hid']} {winner['name']} (cota {winner['bound']:.4f})")

    final = winner["factory"]()                   # reentrenar con TODOS los datos
    final.fit(X, y)
    bundle = {
        "model": final,
        "feature_names": list(FEATURE_NAMES),
        "hierarchy_id": winner["hid"],
        "hierarchy_name": winner["name"],
        "empirical_error": winner["err"],
        "penalty": winner["bound"] - winner["err"],
        "bound": winner["bound"],
        "n_samples": n,
        "n_positive": int((y == 1).sum()),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with args.out.open("wb") as fh:
        pickle.dump(bundle, fh)
    print(f"[SRM] modelo guardado: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
