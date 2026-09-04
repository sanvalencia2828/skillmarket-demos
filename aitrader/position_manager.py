"""
position_manager.py
Gestiona el estado de las posiciones abiertas en Paper Trading.
Usa escritura atómica (os.replace) para concurrencia segura.
"""
import json
import os
import time
import tempfile
import logging

STATE_FILE = "bot_state.json"

def _read_state():
    """Lee el estado actual de forma segura."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logging.warning(f"[PositionManager] Error leyendo estado: {e}. Devolviendo vacío.")
        return {}

def _atomic_write_state(state):
    """Escribe el estado de forma atómica para evitar corrupción."""
    try:
        # Escribir en un archivo temporal primero
        fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE) or '.', suffix='.json')
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=4)
        # Reemplazar el archivo original atómicamente
        os.replace(tmp_path, STATE_FILE)
    except Exception as e:
        logging.error(f"[PositionManager] Fallo crítico escribiendo estado: {e}")

def get_position(token_address):
    """Obtiene los datos de una posición abierta por su dirección."""
    state = _read_state()
    return state.get(token_address) # Devuelve dict o None

def open_position(token_address, entry_price, tp, sl):
    """Registra una nueva posición abierta."""
    state = _read_state()
    state[token_address] = {
        "entry_price": entry_price,
        "last_price": entry_price,   # Para calcular el PnL de la siguiente vela
        "tp": tp,
        "sl": sl,
        "entry_time": time.time(),
        "hold_steps": 0              # Contador de velas mantenidas
    }
    _atomic_write_state(state)

def update_position_price(token_address, current_price):
    """Actualiza el last_price y suma 1 a hold_steps (se llama en cada vela de HOLD)."""
    state = _read_state()
    if token_address in state:
        state[token_address]["last_price"] = current_price
        state[token_address]["hold_steps"] = state[token_address].get("hold_steps", 0) + 1
        _atomic_write_state(state)
        return True
    return False

def close_position(token_address):
    """Elimina la posición del estado (se llama al hacer EXIT)."""
    state = _read_state()
    if token_address in state:
        del state[token_address]
        _atomic_write_state(state)
        return True
    return False
