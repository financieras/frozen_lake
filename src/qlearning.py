import numpy as np
from collections import defaultdict
from typing import Tuple, Dict
import pickle
from pathlib import Path
from tqdm import tqdm

class QLearningAgent:
    def __init__(self, env, params: Dict):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            env: Entorno FrozenLake
            params: Diccionario con hiperparámetros:
                - alpha: Tasa de aprendizaje
                - gamma: Factor de descuento
                - epsilon: Probabilidad de exploración
                - epsilon_decay: Decaimiento de epsilon
                - min_epsilon: Valor mínimo de epsilon
        """
        self.env = env
        self.alpha = params['alpha']
        self.gamma = params['gamma']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.min_epsilon = params['min_epsilon']
        
        # Inicializar Q-table como defaultdict de numpy arrays
        self.q_table = defaultdict(lambda: np.zeros(len(self.env.actions)))
        
    def get_action(self, state: Tuple[int, int]) -> int:
        """
        Selecciona acción usando política ε-greedy.
        
        Args:
            state: Tupla (fila, columna) representando la posición actual
            
        Returns:
            Índice de la acción a tomar (0-3)
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(len(self.env.actions))  # Exploración
        else:
            return np.argmax(self.q_table[state])  # Explotación
    
    def update_q_table(self, state: Tuple[int, int], action: int, 
                      reward: float, next_state: Tuple[int, int]) -> None:
        """
        Actualiza la Q-table usando la ecuación de Bellman.
        """
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
    
    def train(self, episodes: int, show_progress: bool = True) -> Dict:
        """
        Entrena al agente sobre múltiples episodios.
        
        Args:
            episodes: Número de episodios de entrenamiento
            show_progress: Mostrar barra de progreso
            
        Returns:
            Diccionario con historial de recompensas y pasos por episodio
        """
        stats = {'rewards': np.zeros(episodes), 'steps': np.zeros(episodes)}
        
        episode_range = tqdm(range(episodes), disable=not show_progress)
        for episode in episode_range:
            state = self.env.reset()
            done = False
            total_reward = 0
            step = 0
            
            while not done:
                action = self.get_action(state)
                next_state, reward, done = self.env.step(action, state)
                self.update_q_table(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
                step += 1
            
            # Decaimiento de epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Registrar estadísticas
            stats['rewards'][episode] = total_reward
            stats['steps'][episode] = step
            
            if show_progress:
                episode_range.set_description(
                    f"Recomp. media: {stats['rewards'][:episode+1].mean():.2f} "
                    f"Epsilon: {self.epsilon:.3f}"
                )
        
        return stats
    
    def save_model(self, filepath: str) -> None:
        """
        Guarda la Q-table en un archivo.
        """
        Path(filepath).parent.mkdir(exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filepath: str) -> None:
        """
        Carga la Q-table desde un archivo.
        """
        with open(filepath, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(len(self.env.actions)), pickle.load(f))