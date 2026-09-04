"""
trading_env.py
Entorno Gymnasium para AI TRADER.
Estandariza el MDP: observation_space, action_space, step(), reset().
Compatible con stable-baselines3 en el futuro (PPO, SAC).

Contrato con los datos reales del proyecto:
- El espacio de features por defecto es paper_logger.FEATURE_NAMES (22) — las
  mismas claves que el Shadow Mode guarda en rl_transition.state. La lista
  raw de trending (volume_24h, holder_count, ...) NO coincide con el estado
  guardado y daría observaciones en cero.
- state_dim = 22 features de mercado + 3 de posición (is_holding,
  unrealized_pnl, dist_to_tp) = 25.
- Nota: las transiciones guardadas no incluyen 'price' (solo features), por
  lo que en backtest el bookkeeping de posición usa price=0 con guards.
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:
    raise ImportError("gymnasium no instalado: pip install gymnasium") from e

try:
    import paper_logger as _plog
    DEFAULT_FEATURES = list(_plog.FEATURE_NAMES)   # 22, unica fuente de verdad
except Exception:                                  # degradacion: lista literal
    DEFAULT_FEATURES = [
        'volatility', 'smart_money_netflow', 'holder_pnl_ratio', 'trader_buy_ratio',
        'chg_1h', 'chg_5m', 'buy_ratio', 'turnover', 'liquidity', 'mcap', 'age_min',
        'bundler', 'dev_hold', 'top10', 'buy_tax', 'sell_tax', 'rug_ratio',
        'smart_degen', 'renowned', 'sm_confluence', 'sniper_count', 'dev_score',
    ]


class TradingEnv(gym.Env):
    """
    Entorno de trading para micro-caps.

    Modos:
    - 'backtest': Lee transiciones históricas de paper_trades.jsonl
    - 'live': En producción, se le inyectan features en tiempo real (inject_live_state)

    Metadata (para el gate del ecosistema):
    - action_space: Discrete(4) -> {0: 'ENTER', 1: 'EXIT', 2: 'HOLD', 3: 'SKIP'}
      (mismo orden que ACTION_MAP de train_dqn.py; WATCH se normaliza a SKIP aguas arriba)
    - info['valid_actions']: máscara informativa (0/3 sin posición; 1/2 con posición)
      para wrappers tipo MaskablePPO (sb3-contrib) si se desea aplicar dura.
    """

    # Metadatos estándar de Gymnasium
    metadata = {"render_modes": ["human"]}

    ACTION_MAP = {0: 'ENTER', 1: 'EXIT', 2: 'HOLD', 3: 'SKIP'}

    def __init__(self, mode='backtest', jsonl_path=None,
                 feature_names=None, max_steps=100):
        super(TradingEnv, self).__init__()

        self.mode = mode
        self.jsonl_path = pathlib.Path(jsonl_path) if jsonl_path else (ROOT / "outputs" / "paper_trades.jsonl")
        self.max_steps = max_steps
        self.current_step = 0

        # Features de mercado + 3 de posición = dimensión del estado
        self.feature_names = list(feature_names) if feature_names else list(DEFAULT_FEATURES)

        state_dim = len(self.feature_names) + 3  # +3: is_holding, unrealized_pnl, dist_to_tp

        # --- Espacios de Gymnasium ---
        # Observación: vector continuo, sanitizado (NaN/inf -> limites)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )

        # Acción: discreta (0=ENTER, 1=EXIT, 2=HOLD, 3=SKIP)
        self.action_space = spaces.Discrete(4)

        # Estado interno
        self.position = None  # None si no hay posición, dict si la hay
        self.transitions = []
        self.current_idx = 0
        self._live_obs = None

        if mode == 'backtest':
            self._load_historical_data()

    def _load_historical_data(self):
        """Carga transiciones del JSONL para modo backtest."""
        if not self.jsonl_path.exists():
            print(f"[TradingEnv] No se encontró {self.jsonl_path}")
            return

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get('kind') == 'rl_transition':
                        self.transitions.append(data)
                except json.JSONDecodeError:
                    continue

        print(f"[TradingEnv] Cargadas {len(self.transitions)} transiciones históricas.")

    def _build_observation(self, feature_dict):
        """Construye el vector de observación desde un dict de features."""
        # Features de mercado (orden fijo = FEATURE_NAMES)
        market_features = []
        for name in self.feature_names:
            val = feature_dict.get(name, 0.0)
            try:
                market_features.append(float(val))
            except (ValueError, TypeError):
                market_features.append(0.0)

        # Features de posición
        if self.position:
            current_price = float(feature_dict.get('price', 0.0) or 0.0)
            entry_price = float(self.position.get('entry_price', 0.0) or 0.0)
            tp = float(self.position.get('tp', 0.0) or 0.0)

            unrealized_pnl = (current_price - entry_price) / entry_price if entry_price > 0 else 0.0
            dist_to_tp = (tp - current_price) / current_price if (tp > 0 and current_price > 0) else 0.0
            is_holding = 1.0
        else:
            unrealized_pnl = 0.0
            dist_to_tp = 0.0
            is_holding = 0.0

        obs = np.array(market_features + [is_holding, unrealized_pnl, dist_to_tp], dtype=np.float32)

        # Sanitizar NaNs e Infs
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs

    def _get_valid_actions(self):
        """Devuelve las acciones válidas según el estado de la posición."""
        if self.position is None:
            return [0, 3]  # ENTER, SKIP
        else:
            return [1, 2]  # EXIT, HOLD

    def reset(self, seed=None, options=None):
        """Reinicia el entorno para un nuevo episodio."""
        super().reset(seed=seed)

        self.position = None
        self.current_step = 0
        self.current_idx = 0

        if self.mode == 'backtest' and self.transitions:
            # Empezar desde una transición aleatoria
            self.current_idx = int(np.random.randint(0, max(1, len(self.transitions) - self.max_steps)))
            obs = self._build_observation(self.transitions[self.current_idx]['state'])
        else:
            # Estado vacío en modo live (hasta inject_live_state)
            obs = self._build_observation({})

        info = {'valid_actions': self._get_valid_actions()}
        return obs, info

    def step(self, action):
        """
        Ejecuta una acción en el entorno.

        Returns:
            observation (np.array): Nuevo estado
            reward (float): Recompensa obtenida
            terminated (bool): Si el episodio terminó (EXIT o SL/TP)
            truncated (bool): Si se cortó por max_steps
            info (dict): Información adicional
        """
        reward = 0.0
        terminated = False
        truncated = False
        info = {'valid_actions': self._get_valid_actions()}

        # Mapear acción (mismo orden que ACTION_MAP de train_dqn.py)
        action_name = self.ACTION_MAP.get(int(action), 'SKIP')

        # --- Lógica de transición ---
        if self.mode == 'backtest' and self.transitions:
            # Usar la recompensa guardada en el JSONL
            trans = self.transitions[self.current_idx]
            reward = float(trans.get('reward', 0.0) or 0.0)

            # Actualizar posición interna
            if action_name == 'ENTER' and self.position is None:
                state = trans.get('state', {})
                price = float(state.get('price', 0.0) or 0.0)
                self.position = {
                    'entry_price': price,
                    'tp': price * 1.6,
                    'sl': price * 0.65,
                    'last_price': price,
                    'hold_steps': 0
                }
            elif action_name == 'HOLD' and self.position:
                state = trans.get('state', {})
                self.position['last_price'] = float(state.get('price', self.position['last_price']) or 0.0)
                self.position['hold_steps'] += 1
            elif action_name == 'EXIT':
                terminated = True
                self.position = None

            # Avanzar al siguiente estado
            self.current_idx += 1
            if self.current_idx >= len(self.transitions):
                terminated = True
                self.current_idx = 0

            next_state = self.transitions[self.current_idx]['state']
            obs = self._build_observation(next_state)
        else:
            # Modo live: obs inyectada con inject_live_state (si hay); si no, vacía
            if self._live_obs is not None:
                obs = self._live_obs
                self._live_obs = None          # se consume tras cada step
            else:
                obs = self._build_observation({})

        # Truncar si se excede el máximo de pasos
        self.current_step += 1
        if self.current_step >= self.max_steps:
            truncated = True
            if self.position:
                self.position = None

        info['valid_actions'] = self._get_valid_actions()

        return obs, reward, terminated, truncated, info

    def inject_live_state(self, feature_dict, current_reward=0.0, is_done=False):
        """
        En modo live, inyecta el estado actual del mercado.
        El agente llama a step() después, y este método le provee el obs.
        """
        self._live_obs = self._build_observation(feature_dict)
        self._live_reward = current_reward
        self._live_done = is_done

    def render(self):
        """Imprime el estado actual (para debug)."""
        if self.position:
            print(f"[Env] Holding | Entry: {self.position['entry_price']:.6f} | Steps: {self.position['hold_steps']}")
        else:
            print(f"[Env] No position | Step: {self.current_step}/{self.max_steps}")
