"""q_learning_offline.py — Entrenamiento offline de Tabular Q-Learning.

Lee las transiciones rl_transition guardadas por el Shadow Mode en
outputs/paper_trades.jsonl, reconstruye la secuencia temporal POR TOKEN
(next_state = estado de la siguiente transicion del MISMO token) y
actualiza la Q-table con la ecuacion de Bellman:

    Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

Robusto: archivo inexistente / vacio / con pocas muestras / lineas
corruptas -> mensaje y salida limpia, nunca crash.

Uso:  python q_learning_offline.py
"""

import json
import pickle
import pathlib
import sys
from collections import defaultdict

import numpy as np

# Ruta relativa al script (no al CWD) para que funcione desde cualquier dir
ROOT = pathlib.Path(__file__).parent
JSONL_PATH = ROOT / "outputs" / "paper_trades.jsonl"
QTABLE_PATH = ROOT / "q_table.pkl"

# 1. Cargar y encadenar transiciones por token
# Como guardamos next_state=None en vivo, aqui reconstruimos la secuencia temporal
raw_transitions = []
try:
    with JSONL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if data.get("kind") == "rl_transition":
                    raw_transitions.append(data)
            except json.JSONDecodeError:
                continue
except FileNotFoundError:
    print(f"Sin dataset RL: {JSONL_PATH} no existe. Ejecuta el bot en LIVE más tiempo.")
    sys.exit(1)

if not raw_transitions:
    print("No hay transiciones RL todavía. Ejecuta el bot en LIVE más tiempo.")
    sys.exit(1)

if len(raw_transitions) < 10:
    print(f"Solo {len(raw_transitions)} transiciones (se recomiendan >10 para Q-learning). "
          "Entrenando de todos modos...")

# Ordenar por token y timestamp para encadenar estados.
# token_address viene del registro (el state dict de features NO lo contiene);
# registros antiguos sin token_address caen en '' y se encadenan entre si por ts.
raw_transitions.sort(key=lambda x: (x.get("token_address", ""), x["timestamp"]))

transitions = []
for i, trans in enumerate(raw_transitions):
    # Si la transaccion actual no es terminal, el next_state es el estado de la
    # siguiente linea DEL MISMO TOKEN (si cambia el token, la secuencia termina)
    if (i + 1 < len(raw_transitions)
            and not trans["done"]
            and raw_transitions[i + 1].get("token_address", "") == trans.get("token_address", "")):
        trans["next_state"] = raw_transitions[i + 1]["state"]
    else:
        trans["next_state"] = None
    transitions.append(trans)

print(f"Cargadas y encadenadas {len(transitions)} transiciones.")


# 2. Funcion de discretizacion ajustada a las features reales
def discretize(state_dict):
    if not state_dict:
        return "terminal"

    # Usamos .get() con defaults seguros por si falta alguna feature
    sm_netflow = state_dict.get("smart_money_netflow", 0)
    vol_5m = state_dict.get("volatility", 0)  # o chg_5m si es lo que guardas
    is_holding = state_dict.get("is_holding", 0)  # no vive en state (vive en position_manager); 0 por ahora

    # Buckets simples para reducir el espacio de estados
    sm_bucket = 1 if sm_netflow > 0 else (-1 if sm_netflow < 0 else 0)
    vol_bucket = 1 if vol_5m > 0.1 else (0 if vol_5m > 0 else -1)

    return f"sm_{sm_bucket}_vol_{vol_bucket}_hold_{is_holding}"


# 3. Inicializar Q-table
actions = ["ENTER", "EXIT", "HOLD", "SKIP"]
Q = defaultdict(lambda: {a: 0.0 for a in actions})

# 4. Hiperparametros
alpha = 0.1   # Learning rate
gamma = 0.9   # Discount factor

# 5. Bucle de actualizacion (offline)
for trans in transitions:
    s = discretize(trans["state"])
    a = trans["action"]
    if a not in Q[s]:                          # accion inesperada -> fila nueva segura
        Q[s] = {act: 0.0 for act in actions}
    r = trans["reward"]
    done = trans["done"]
    ns = discretize(trans["next_state"]) if trans["next_state"] else "terminal"

    # Si es terminal, no hay max_next_q
    max_next_q = max(Q[ns].values()) if ns != "terminal" else 0.0

    current_q = Q[s][a]
    Q[s][a] = current_q + alpha * (r + gamma * max_next_q - current_q)

# 6. Guardar Q-table (para consumo futuro; el blend en ScoringEngine queda como TODO)
try:
    with QTABLE_PATH.open("wb") as fh:
        pickle.dump({"q_table": dict(Q), "actions": actions,
                     "alpha": alpha, "gamma": gamma,
                     "n_transitions": len(transitions)}, fh)
    print(f"\nQ-table guardada: {QTABLE_PATH}")
except Exception as e:
    print(f"\nNo se pudo guardar la Q-table: {e}")

# 7. Mostrar politica aprendida
print("\nPolítica Óptima Aprendida (Top estados):")
for i, (state, q_vals) in enumerate(Q.items()):
    if i > 15:
        break
    best_action = max(q_vals, key=q_vals.get)
    print(f"Estado {state}: Mejor Acción = {best_action} (Q={max(q_vals.values()):.4f})")

# TODO futuro: integrar la Q-table en el ScoringEngine para consumo en vivo
