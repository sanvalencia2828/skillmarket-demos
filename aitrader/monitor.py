#!/usr/bin/env python3
"""
monitor.py - Panel de control para el agente RL de AI TRADER.
Ejecuta: python monitor.py
Para actualización automática: python monitor.py --watch (refresca cada 5 segundos)
"""

import json
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

# Windows: consola cp1252 crashea con emojis/box-drawing sin esto
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --- Configuración (rutas relativas al script, no al CWD) ---
ROOT = Path(__file__).parent
JSONL_PATH = ROOT / "outputs" / "paper_trades.jsonl"
MODEL_PATH = ROOT / "dqn_model.pkl"
META_PATH = ROOT / "dqn_meta.json"
GATE_THRESHOLD = 500

# --- Colores ANSI para la terminal ---
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def clear_screen():
    """Limpia la pantalla para el modo watch."""
    os.system('cls' if os.name == 'nt' else 'clear')

def count_transitions():
    """Cuenta y analiza las transiciones RL en el JSONL."""
    if not JSONL_PATH.exists():
        return None, {"error": "Archivo no encontrado"}

    total_lines = 0
    rl_transitions = []
    actions = Counter()
    rewards = []
    tokens = set()
    has_token_address = False
    last_transition_time = None
    state_features_count = Counter()

    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            try:
                data = json.loads(line)
                if data.get('kind') == 'rl_transition':
                    rl_transitions.append(data)
                    # Acción
                    act = data.get('action')
                    if act:
                        actions[act] += 1

                    # Recompensa
                    r = data.get('reward')
                    if r is not None:
                        rewards.append(float(r))

                    # Token address
                    token = data.get('token_address')
                    if token:
                        tokens.add(token)
                        has_token_address = True

                    # Timestamp
                    ts = data.get('timestamp')
                    if ts:
                        last_transition_time = datetime.fromtimestamp(ts)

                    # Contar features del estado (para ver consistencia)
                    state = data.get('state', {})
                    if isinstance(state, dict):
                        for k in state.keys():
                            state_features_count[k] += 1

            except json.JSONDecodeError:
                continue

    n_transitions = len(rl_transitions)
    gate_status = "ABIERTO" if n_transitions >= GATE_THRESHOLD else "CERRADO"

    return {
        "total_lines": total_lines,
        "n_transitions": n_transitions,
        "gate_status": gate_status,
        "actions": dict(actions),
        "rewards": rewards,
        "tokens_count": len(tokens),
        "has_token_address": has_token_address,
        "last_transition_time": last_transition_time,
        "state_features_count": dict(state_features_count.most_common(5)),  # top 5
    }, rl_transitions

def check_model():
    """Verifica si el modelo DQN existe y si el gate lo abriría."""
    model_exists = MODEL_PATH.exists()
    meta_exists = META_PATH.exists()

    if meta_exists:
        try:
            with open(META_PATH, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            trained_transitions = meta.get('n_transitions', 0)
            trained_at = meta.get('trained_at', 'desconocido')
        except Exception:
            trained_transitions = 0
            trained_at = 'error'
    else:
        trained_transitions = 0
        trained_at = 'no entrenado'

    return {
        "model_exists": model_exists,
        "meta_exists": meta_exists,
        "trained_transitions": trained_transitions,
        "trained_at": trained_at,
        "ready_to_activate": model_exists and trained_transitions >= GATE_THRESHOLD
    }

def format_reward_stats(rewards):
    """Calcula estadísticas básicas de recompensa."""
    if not rewards:
        return None

    arr = np.array(rewards)
    positive = np.sum(arr > 0)
    negative = np.sum(arr < 0)
    zero = np.sum(arr == 0)

    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "positive": int(positive),
        "negative": int(negative),
        "zero": int(zero),
        "total": len(arr)
    }

def display_dashboard(data, model_info, reward_stats):
    """Imprime el dashboard en formato bonito."""
    clear_screen()
    print(f"{Colors.BOLD}{Colors.CYAN}╔═══════════════════════════════════════════════════════════╗{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}║         AI TRADER - RL MONITOR                          ║{Colors.END}")
    print(f"{Colors.CYAN}╚═══════════════════════════════════════════════════════════╝{Colors.END}")
    print(f"    Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("")

    # --- Sección 1: Transiciones y Gate ---
    print(f"{Colors.BOLD}📊 DATOS RECOLECTADOS{Colors.END}")
    print(f"  Transiciones RL: {data['n_transitions']} / {GATE_THRESHOLD} (umbral)")

    # Barra de progreso
    progress = min(data['n_transitions'] / GATE_THRESHOLD, 1.0)
    bar_len = 30
    filled = int(bar_len * progress)
    bar = '█' * filled + '░' * (bar_len - filled)
    print(f"  [{bar}] {progress*100:.1f}%")

    if data['n_transitions'] >= GATE_THRESHOLD:
        print(f"  {Colors.GREEN}✅ Gate: {Colors.BOLD}ABIERTO{Colors.END} (DQN se activará en scoring.py)")
    else:
        faltan = GATE_THRESHOLD - data['n_transitions']
        print(f"  {Colors.YELLOW}⏳ Gate: CERRADO (faltan {faltan} transiciones){Colors.END}")
    print("")

    # --- Sección 2: Modelo DQN ---
    print(f"{Colors.BOLD}🧠 MODELO DQN{Colors.END}")
    if model_info['model_exists']:
        print(f"  {Colors.GREEN}✅ Modelo encontrado{Colors.END}")
        print(f"  Entrenado con: {model_info['trained_transitions']} transiciones")
        print(f"  Fecha: {model_info['trained_at']}")
        if model_info['ready_to_activate']:
            print(f"  {Colors.GREEN}✅ Listo para activar (gate superado){Colors.END}")
        else:
            print(f"  {Colors.YELLOW}⏳ Esperando gate (necesita {GATE_THRESHOLD} transiciones){Colors.END}")
    else:
        print(f"  {Colors.RED}❌ Modelo no encontrado (ejecuta train_dqn.py){Colors.END}")
    print("")

    # --- Sección 3: Estadísticas de Acciones ---
    print(f"{Colors.BOLD}🎮 DISTRIBUCIÓN DE ACCIONES{Colors.END}")
    if data['actions']:
        total_acts = sum(data['actions'].values())
        for act, count in sorted(data['actions'].items(), key=lambda x: -x[1]):
            pct = (count / total_acts) * 100
            print(f"  {act:<8}: {count:>4} ({pct:>5.1f}%)")
    else:
        print(f"  {Colors.YELLOW}Sin acciones registradas{Colors.END}")
    print("")

    # --- Sección 4: Recompensas ---
    print(f"{Colors.BOLD}💰 ESTADÍSTICAS DE RECOMPENSA{Colors.END}")
    if reward_stats and reward_stats['total'] > 0:
        print(f"  Media: {reward_stats['mean']:+.4f}")
        print(f"  Desv. Estandar: {reward_stats['std']:.4f}")
        print(f"  Mín/Máx: {reward_stats['min']:.4f} / {reward_stats['max']:.4f}")
        print(f"  Positivas: {reward_stats['positive']}  |  Negativas: {reward_stats['negative']}  |  Cero: {reward_stats['zero']}")
        # Ratio de victorias (win rate)
        win_rate = reward_stats['positive'] / reward_stats['total'] if reward_stats['total'] > 0 else 0
        print(f"  Win Rate: {win_rate*100:.1f}%")
    else:
        print(f"  {Colors.YELLOW}Sin recompensas registradas aún{Colors.END}")
    print("")

    # --- Sección 5: Metadata ---
    print(f"{Colors.BOLD}📌 METADATA{Colors.END}")
    print(f"  Tokens únicos: {data['tokens_count']}")
    print(f"  Tiene 'token_address': {'✅' if data['has_token_address'] else '❌'}")
    if data['last_transition_time']:
        print(f"  Última transición: {data['last_transition_time'].strftime('%H:%M:%S')}")
    print(f"  Líneas totales en JSONL: {data['total_lines']}")
    print("")

    # --- Sección 6: Top Features en estado ---
    print(f"{Colors.BOLD}🔍 TOP 5 FEATURES EN ESTADO{Colors.END}")
    for feature, count in data['state_features_count'].items():
        print(f"  {feature}: {count}")
    print("")

    print(f"{Colors.CYAN}─────────────────────────────────────────────────────────────{Colors.END}")
    print(f"{Colors.YELLOW}Presiona Ctrl+C para salir.  --watch para refresco automático.{Colors.END}")

def main():
    watch_mode = '--watch' in sys.argv or '-w' in sys.argv

    try:
        while True:
            data, _ = count_transitions()
            if data is None:
                print(f"{Colors.RED}Error: No se pudo leer {JSONL_PATH}{Colors.END}")
                if watch_mode:
                    # el archivo aparecera cuando el bot escriba la primera transicion
                    print(f"{Colors.YELLOW}Esperando que el bot cree el JSONL... (Ctrl+C para salir){Colors.END}")
                    time.sleep(5)
                    continue
                break

            model_info = check_model()
            reward_stats = format_reward_stats(data.get('rewards', []))

            display_dashboard(data, model_info, reward_stats)

            if not watch_mode:
                break

            time.sleep(5)  # Refresco cada 5 segundos

    except KeyboardInterrupt:
        print(f"\n{Colors.GREEN}Monitor detenido.{Colors.END}")

if __name__ == "__main__":
    main()
