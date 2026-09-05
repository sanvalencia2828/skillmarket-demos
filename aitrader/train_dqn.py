"""train_dqn.py — Entrenamiento OFFLINE del agente DQN (Q-network + Bellman).

Implementa DQN offline via fitted Q-iteration (Ernst et al. 2005):
  - Q-network: MLPRegressor (sklearn, ya en requirements; sin torch —
    sustituible por un QNet de torch cuando el dataset lo justifique)
  - Targets Bellman:  y = r + gamma * (1-done) * max_a' Q(s', a')
  - Itera: predecir targets con la red actual -> refit -> repetir

Entrada: transiciones rl_transition del Shadow Mode en outputs/paper_trades.jsonl
Salida:  dqn_model.pkl (red Q) + dqn_meta.json (metadata barata para el gate
         del ScoringEngine: solo consume el modelo en vivo con >= GATE_TRANSITIONS)

Contrato con los datos reales (verificado contra el JSONL):
  - action se guarda como STRING ('ENTER'/'WATCH'/'SKIP'/'EXIT') -> ACTION_MAP
    string->int; WATCH se normaliza a SKIP (mismo significado economico).
  - state NO es uniforme: transiciones de scoring.py traen las 22 claves de
    FEATURE_NAMES; las del bridge de exits traen solo {"price": ...} ->
    build_state_vector rellena faltantes con defaults (dev_score -> 0.5).
  - token_address puede ser None (registros viejos) -> episodio de un paso.

Robusto: JSONL inexistente/vacio/con pocas muestras/lineas corruptas/acciones
desconocidas -> aviso y salida limpia, nunca crash.
Uso:  python train_dqn.py
Paths sobre-escribibles por env: DQN_JSONL, DQN_MODEL_OUT
"""

import json
import os
import pathlib
import pickle
import sys
import datetime

import numpy as np

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_logger import FEATURE_NAMES
import paper_logger as _plog

JSONL_PATH = pathlib.Path(os.environ.get("DQN_JSONL", str(ROOT / "outputs" / "paper_trades.jsonl")))
MODEL_PATH = pathlib.Path(os.environ.get("DQN_MODEL_OUT", str(ROOT / "dqn_model.pkl")))
META_PATH = MODEL_PATH.with_name("dqn_meta.json")   # scoring.py y monitor.py leen ESTE nombre

MIN_SAMPLES = 100        # umbral de aviso "faltan datos" (entrena igual)
GATE_TRANSITIONS = 500   # el ScoringEngine solo consume en vivo con >= este numero

# JSONL guarda action como STRING; WATCH -> SKIP (mismo significado economico)
ACTION_MAP = {"ENTER": 0, "EXIT": 1, "HOLD": 2, "SKIP": 3}
ACTIONS = ["ENTER", "EXIT", "HOLD", "SKIP"]

GAMMA = 0.9
ITERATIONS = 12          # pasadas de fitted Q-iteration
HIDDEN = (64, 64)
SEED = 42
DOUBLE_DQN = os.environ.get("DQN_DOUBLE", "1").strip().lower() in ("1", "true", "yes", "on")
# seleccion con la red principal, evaluacion con la target congelada
# (reduce el sesgo de sobreestimacion del max de Bellman; desactivable con DQN_DOUBLE=0)


def build_state_vector(state_dict) -> np.ndarray:
    """dict de features -> np.array(22) en el orden EXACTO de FEATURE_NAMES.

    Delega en paper_logger.vectorize (unica fuente de verdad, usa los mismos
    nombres y el mismo orden que el entrenamiento SRM): tolera claves
    faltantes (0.0; dev_score None -> 0.5 NEUTRAL), bools, NaN/inf, strings,
    y el state fallback {"price": ...} del bridge (todo a defaults)."""
    if not isinstance(state_dict, dict):
        state_dict = {}
    return np.asarray(_plog.vectorize(state_dict), dtype=np.float32)


def action_index(action):
    """String del JSONL -> indice del action space (WATCH -> SKIP)."""
    if action == "WATCH":
        action = "SKIP"
    return ACTION_MAP.get(action)


def load_transitions():
    """Lee rl_transitions del JSONL. Devuelve None si el archivo no existe."""
    raw = []
    try:
        with JSONL_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue                     # linea corrupta se salta
                if d.get("kind") == "rl_transition":
                    raw.append(d)
    except FileNotFoundError:
        return None
    return raw


def chain_episodes(raw: list) -> list:
    """Ordena por (token_address, timestamp) y encadena next_state POR TOKEN.

    - transicion no-done con siguiente registro del MISMO token -> next_state
    - done, ultimo del token, o token_address None -> next_state=None (terminal)
    """
    raw = sorted(raw, key=lambda x: (x.get("token_address") or "", x.get("timestamp") or 0.0))
    episodes = []
    for i, t in enumerate(raw):
        addr = t.get("token_address")
        nxt = None
        if (not t.get("done")) and addr is not None and i + 1 < len(raw):
            nt = raw[i + 1]
            if nt.get("token_address") == addr:
                nxt = nt.get("state")
        episodes.append((t.get("state") or {}, t.get("action"),
                         float(t.get("reward") or 0.0), nxt, bool(t.get("done"))))
    return episodes


def build_dataset(episodes: list):
    """Episodios -> (X, A, R, Xn, DONE, skipped_actions)."""
    X, A, R, Xn, DONE, skipped = [], [], [], [], [], []
    for state, action, reward, next_state, done in episodes:
        ai = action_index(action)
        if ai is None:
            skipped.append(str(action))
            continue
        X.append(build_state_vector(state))
        A.append(ai)
        R.append(reward)
        Xn.append(build_state_vector(next_state) if next_state else None)
        DONE.append(1.0 if done else 0.0)
    X = np.asarray(X, dtype=np.float32) if X else np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return (X, np.asarray(A, dtype=np.int64), np.asarray(R, dtype=np.float32), Xn,
            np.asarray(DONE, dtype=np.float32), skipped)


def _q_all(net, states: np.ndarray) -> np.ndarray:
    """Q(s', a') para TODAS las acciones de cada estado -> matriz (n_estados, n_acciones)."""
    reps = np.repeat(states, len(ACTIONS), axis=0)
    acts = np.tile(np.eye(len(ACTIONS), dtype=np.float32), (len(states), 1))
    q = net.predict(np.hstack([reps, acts]).astype(np.float32))
    return q.reshape(len(states), len(ACTIONS))


def _q_max(net, states: np.ndarray) -> np.ndarray:
    """Vanilla DQN: max_a' Q(s', a') — la MISMA red selecciona y evalúa."""
    return _q_all(net, states).max(axis=1)


def _q_eval(net, states: np.ndarray, action_idx: np.ndarray) -> np.ndarray:
    """Q(s', a_elegida) — evalúa SOLO la acción indicada (gather del Double DQN)."""
    acts = np.eye(len(ACTIONS), dtype=np.float32)[action_idx]
    return net.predict(np.hstack([states, acts]).astype(np.float32))


def train_dqn(X, A, R, Xn, DONE):
    """Fitted Q-iteration con Double DQN (offline).

    Equivalente batch del bloque online:
        best_actions = q_network(next_states).argmax(dim=1)      # seleccionar
        max_next_q   = target_network(next_states).gather(1, best_actions)  # evaluar
        target_q     = rewards + gamma * max_next_q * (~dones)

    En batch: select_net = red del fit más reciente; target_net = copia congelada
    de la generación anterior (retraso de 1 iteracion). it0 sin redes (target=r);
    it1 vanilla (solo hay una red); it2+ Double DQN.

    Q(s,a) se modela como red s|a_onehot -> Q escalar (cabeza compartida).
    Los estados se escalan (StandardScaler) — features como mcap/liquidity
    viven en escala de millones y sin escalar el MLP diverge."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler

    n = len(X)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    Xn_filled = np.array([xn if xn is not None else np.zeros(X.shape[1], dtype=np.float32)
                          for xn in Xn], dtype=np.float32)
    Xn_s = scaler.transform(Xn_filled).astype(np.float32)
    has_next = np.array([xn is not None for xn in Xn], dtype=np.float32)
    A_onehot = np.eye(len(ACTIONS), dtype=np.float32)[A]
    Xsa = np.hstack([Xs, A_onehot]).astype(np.float32)

    select_net = None        # red principal (la mas reciente que aprende)
    target_net = None        # target network (congelada, una generacion atras)
    final_loss = float("nan")
    for it in range(ITERATIONS):
        if select_net is None:
            q_next = np.zeros(n, dtype=np.float32)
        elif target_net is None:
            # solo existe una red: max clasico (no se puede desacoplar todavia)
            q_next = _q_max(select_net, Xn_s)
        elif DOUBLE_DQN:
            # --- DOUBLE DQN ---
            # 1. Seleccionar la mejor accion con la Q-Network principal (la que aprende)
            best_actions = np.argmax(_q_all(select_net, Xn_s), axis=1)
            # 2. Evaluar el Q-value de ESA accion con la Target Network (estable)
            q_next = _q_eval(target_net, Xn_s, best_actions)
        else:
            q_next = _q_max(select_net, Xn_s)   # vanilla (flag OFF)
        target = (R + GAMMA * (1.0 - DONE) * has_next * q_next).astype(np.float32)
        net = MLPRegressor(hidden_layer_sizes=HIDDEN, activation="relu",
                           max_iter=400, random_state=SEED,
                           warm_start=(select_net is not None))
        net.fit(Xsa, target)
        target_net = select_net                 # la red previa se congela como target
        select_net = net                        # la nueva es la principal
        pred = net.predict(Xsa)
        final_loss = float(np.mean((pred - target) ** 2))
        print(f"[DQN] iter {it + 1}/{ITERATIONS} | mse={final_loss:.6f}"
              + (" [double]" if (DOUBLE_DQN and target_net is not None) else ""))
    return select_net, scaler, final_loss


def greedy(net, scaler, state_vec: np.ndarray):
    """Accion greedy y Q-values de un estado (escala el estado antes de predecir)."""
    s = scaler.transform(state_vec.reshape(1, -1)).astype(np.float32)
    reps = np.repeat(s, len(ACTIONS), axis=0)
    acts = np.eye(len(ACTIONS), dtype=np.float32)
    q = net.predict(np.hstack([reps, acts]).astype(np.float32))
    return ACTIONS[int(np.argmax(q))], q.tolist()


def main() -> int:
    raw = load_transitions()
    if raw is None:
        print(f"Sin dataset RL: {JSONL_PATH} no existe. Ejecuta el bot en LIVE más tiempo.")
        return 1
    if not raw:
        print("No hay transiciones RL todavía. Ejecuta el bot en LIVE más tiempo.")
        return 1
    if len(raw) < MIN_SAMPLES:
        print(f"[DQN] AVISO: faltan datos — {len(raw)}/{MIN_SAMPLES} transiciones. Entrenando igual...")

    episodes = chain_episodes(raw)
    X, A, R, Xn, DONE, skipped = build_dataset(episodes)
    if skipped:
        print(f"[DQN] AVISO: {len(skipped)} transiciones con accion desconocida descartadas: {sorted(set(skipped))}")
    if len(X) == 0:
        print("[DQN] sin muestras validas (todas las acciones desconocidas).")
        return 1

    dist = {ACTIONS[a]: int(int((A == a).sum())) for a in sorted(set(A.tolist()))}
    chained = int(sum(1 for xn in Xn if xn is not None))
    print(f"[DQN] dataset: {len(X)} muestras | encadenadas: {chained} | acciones: {dist}")

    net, scaler, loss = train_dqn(X, A, R, Xn, DONE)

    bundle = {
        "model": net,
        "scaler": scaler,
        "feature_names": list(FEATURE_NAMES),
        "action_map": dict(ACTION_MAP),
        "actions": ACTIONS,
        "gamma": GAMMA,
        "hidden": list(HIDDEN),
        "n_transitions": int(len(X)),
        "final_loss": loss,
        "double_dqn": bool(DOUBLE_DQN),
        "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "algo": "double fitted-q-iteration (offline Double DQN, MLP 64x64 + StandardScaler)",
    }
    with MODEL_PATH.open("wb") as fh:
        pickle.dump(bundle, fh)
    # metadata barata para el gate del ScoringEngine (sin unpicklear sklearn)
    META_PATH.write_text(json.dumps({k: bundle[k] for k in
                                     ("n_transitions", "feature_names", "action_map",
                                      "trained_at", "gamma")}), encoding="utf-8")
    print(f"\n[DQN] modelo guardado: {MODEL_PATH}")
    print(f"[DQN] metadata (gate {GATE_TRANSITIONS}): {META_PATH}")

    print("\nPolitica aprendida (muestra, dataset vs greedy):")
    for i in range(min(8, len(X))):
        g, q = greedy(net, scaler, X[i])
        print(f"  s[{i}] tomada={ACTIONS[A[i]]:<6} greedy={g:<6} Q={[round(v, 4) for v in q]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
