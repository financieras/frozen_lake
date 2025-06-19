# Hiperparámetros Q-Learning
QL_PARAMS = {
    'alpha': 0.1,          # Tasa de aprendizaje
    'gamma': 0.95,         # Factor de descuento
    'epsilon': 0.3,        # Probabilidad de exploración inicial
    'epsilon_decay': 0.995, # Decaimiento de epsilon
    'episodes': 10000,     # Número de episodios de entrenamiento
    'min_epsilon': 0.01    # Mínima exploración permitida
}

# Configuración del entorno
ENV_PARAMS = {
    'slippery': True,      # Superficie resbaladiza
    'slippery_prob': 0.3,  # Probabilidad de resbalón
    'penalty': -0.1,       # Penalización por movimiento
    'hole_penalty': -1,    # Penalización por hoyo
    'goal_reward': 1       # Recompensa por meta
}