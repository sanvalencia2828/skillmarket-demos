"""reward_engine.py — Cerebro matemático de recompensa RL (aislado).

Recompensa densa (pnl por paso - coste de tiempo) + terminal (pnl total
+ bonus TP / penalización SL / penalización timeout) + coste de acción.
Aún sin integrar: módulo standalone, el flujo SRM/FastAPI no lo toca.
"""

import math

LAMBDA_TIME = 0.0005
BONUS_TP = 0.02
PENALTY_SL = 0.05
PENALTY_TIMEOUT = 0.03
COST_ENTER = 0.001
COST_EXIT = 0.001
MAX_HOLD_STEPS = 20

def compute_step_reward(current_price, previous_price, entry_price, action, is_done, tp_price, sl_price, hold_steps):
    action_cost = 0.0
    if action == "ENTER": action_cost = -COST_ENTER
    elif action == "EXIT": action_cost = -COST_EXIT
    
    if action == "SKIP" or entry_price is None:
        return action_cost
        
    if previous_price is None or previous_price == 0:
        previous_price = entry_price
        
    pnl_step = (current_price - previous_price) / previous_price
    dense_reward = pnl_step - LAMBDA_TIME
    
    terminal_reward = 0.0
    if is_done and action == "EXIT":
        total_pnl = (current_price - entry_price) / entry_price
        bonus_tp = BONUS_TP if tp_price is not None and current_price >= tp_price else 0.0
        penalty_sl = -PENALTY_SL if sl_price is not None and current_price <= sl_price else 0.0
        penalty_timeout = -PENALTY_TIMEOUT if hold_steps >= MAX_HOLD_STEPS else 0.0
        terminal_reward = total_pnl + bonus_tp + penalty_sl + penalty_timeout
        
    if action == "ENTER":
        return action_cost
        
    return action_cost + dense_reward + terminal_reward
