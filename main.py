from config.params import QL_PARAMS, ENV_PARAMS
from config.maps import MAPS
from src.environment import FrozenLakeEnv
from src.qlearning import QLearningAgent
from src.visualization import animate_episode

# Inicialización
env = FrozenLakeEnv(map_data=MAPS['4x4'], params=ENV_PARAMS)
agent = QLearningAgent(env, QL_PARAMS)

# Entrenamiento (sin animación para velocidad)
agent.train(show_progress=True)

# Visualización de un episodio
animate_episode(env, agent)