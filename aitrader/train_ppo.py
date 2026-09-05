"""
train_ppo.py - Entrenamiento offline de PPO (Proximal Policy Optimization)
con stable-baselines3 sobre el entorno Gymnasium trading_env.py.

Reemplaza la decision por umbral del DQN: la red de politica (MlpPolicy)
decide directamente ENTER/EXIT/HOLD/SKIP observando el estado del mercado.

Salida: ppo_model.zip + ppo_meta.json (gate del ScoringEngine: solo se
consume en vivo con >= GATE_THRESHOLD transiciones; degradacion elegante
si SB3 no esta instalado o el modelo falla).

Uso:  python train_ppo.py [--timesteps 10000]
"""

import os
import json
import pathlib
import sys
import argparse
from datetime import datetime

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows: consola cp1252 crashea con emojis sin esto
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import paper_logger as _plog
from trading_env import TradingEnv

MODEL_PATH = ROOT / "ppo_model.zip"
META_PATH = ROOT / "ppo_meta.json"
GATE_THRESHOLD = 500
DEFAULT_TIMESTEPS = 10000


def train_ppo(total_timesteps: int = DEFAULT_TIMESTEPS) -> int:
    # 0. SB3 con degradacion elegante (import tardio: si falta, mensaje limpio)
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as e:
        print(f"stable-baselines3 no instalado ({e}): pip install stable-baselines3")
        return 1

    # 1. Verificar si hay suficientes datos
    env = TradingEnv(mode='backtest')
    if len(env.transitions) < GATE_THRESHOLD:
        print(f"⚠️ Necesitas al menos {GATE_THRESHOLD} transiciones. Tienes {len(env.transitions)}.")
        return 1

    # 2. Envolver entorno para SB3
    vec_env = DummyVecEnv([lambda: env])

    # 3. Inicializar PPO (MlpPolicy)
    model = PPO("MlpPolicy", vec_env, verbose=1, learning_rate=3e-4,
                n_steps=2048, batch_size=64, seed=42)

    # 4. Entrenar
    print(f"Iniciando entrenamiento PPO ({total_timesteps} timesteps)...")
    model.learn(total_timesteps=total_timesteps)

    # 5. Guardar modelo y metadata (compatible con el gate de scoring.py)
    model.save(str(MODEL_PATH))
    meta = {
        "n_transitions": len(env.transitions),          # clave del gate (scoring.py)
        "trained_transitions": len(env.transitions),    # alias legible
        "policy": "PPO",
        "algo": "stable-baselines3 PPO (MlpPolicy)",
        "feature_names": list(_plog.FEATURE_NAMES),     # orden del obs (22 de mercado)
        "obs_dim": env.observation_space.shape[0],      # 22 + 3 de posicion
        "action_map": {str(i): a for i, a in env.ACTION_MAP.items()},
        "timesteps": total_timesteps,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"✅ PPO entrenado y guardado en {MODEL_PATH}")
    print(f"   metadata (gate {GATE_THRESHOLD}): {META_PATH}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrenamiento PPO offline")
    ap.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = ap.parse_args()
    sys.exit(train_ppo(args.timesteps))
