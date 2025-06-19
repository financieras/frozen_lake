import numpy as np
from typing import Tuple, Dict

class FrozenLakeEnv:
    def __init__(self, map_data: list[str], params: dict):
        """
        Inicializa el entorno FrozenLake.
        
        Args:
            map_data: Lista de strings representando el mapa (ej: ["S·", "·G"])
            params: Diccionario con parámetros (slippery, rewards, etc.)
        """
        self.map = np.array([list(row) for row in map_data])
        self.rows, self.cols = self.map.shape
        self.slippery = params.get('slippery', False)
        self.slip_prob = params.get('slippery_prob', 0.3)
        
        # Definición de acciones (arriba, abajo, izquierda, derecha)
        self.actions = {
            0: (-1, 0),  # Arriba
            1: (1, 0),   # Abajo
            2: (0, -1),  # Izquierda
            3: (0, 1)    # Derecha
        }
        
        # Encontrar posición inicial (S) y meta (G)
        self.start_pos = tuple(np.argwhere(self.map == 'S')[0])
        self.goal_pos = tuple(np.argwhere(self.map == 'G')[0])
        
        # Recompensas y penalizaciones
        self.rewards = {
            'G': params.get('goal_reward', 1),
            'H': params.get('hole_penalty', -1),
            'move': params.get('penalty', -0.01)
        }

    def reset(self) -> Tuple[int, int]:
        """Reinicia el entorno a la posición inicial."""
        return self.start_pos

    def step(self, action: int, current_pos: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action: Índice de la acción a tomar (0-3)
            current_pos: Posición actual (fila, columna)
            
        Returns:
            (new_pos, reward, done)
        """
        # Aplicar resbalones si está activado
        if self.slippery and np.random.random() < self.slip_prob:
            action = self._apply_slippery(action)
        
        # Calcular nueva posición
        move = self.actions[action]
        new_pos = (current_pos[0] + move[0], current_pos[1] + move[1])
        
        # Verificar límites del mapa
        new_pos = self._check_boundaries(new_pos)
        
        # Obtener tipo de celda
        cell_type = self.map[new_pos[0], new_pos[1]]
        
        # Calcular recompensa y terminación
        reward, done = self._get_reward(cell_type)
        
        return new_pos, reward, done

    def _apply_slippery(self, action: int) -> int:
        """Modifica la acción debido a resbalones."""
        # Direcciones perpendiculares (ej: para arriba -> izquierda/derecha)
        perpendicular_actions = {
            0: [2, 3],  # Arriba -> Izquierda/Derecha
            1: [2, 3],  # Abajo -> Izquierda/Derecha
            2: [0, 1],  # Izquierda -> Arriba/Abajo
            3: [0, 1]   # Derecha -> Arriba/Abajo
        }
        return np.random.choice(perpendicular_actions[action])

    def _check_boundaries(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        """Asegura que la posición no salga del mapa."""
        row = np.clip(pos[0], 0, self.rows - 1)
        col = np.clip(pos[1], 0, self.cols - 1)
        return (row, col)

    def _get_reward(self, cell_type: str) -> Tuple[float, bool]:
        """Determina recompensa y si el episodio termina."""
        if cell_type == 'H':
            return self.rewards['H'], True
        elif cell_type == 'G':
            return self.rewards['G'], True
        else:
            return self.rewards['move'], False

    def is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        """Verifica si una posición es válida (no es un hoyo)."""
        return self.map[pos[0], pos[1]] != 'H'