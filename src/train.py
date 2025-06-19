#!/usr/bin/env python3
"""
Script de entrenamiento para FrozenLake Q-Learning
"""

from qlearning import QLearningAgent
from environment import FrozenLakeEnv
from config.params import QL_PARAMS
from config.maps import MAPS
import argparse

def main():
    # Configuración de argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument('--map', type=str, default='4x4', choices=['4x4', '8x8'])
    parser.add_argument('--episodes', type=int, default=10000)
    parser.add_argument('--slippery', type=int, default=1, choices=[0, 1])
    args = parser.parse_args()

    # Inicialización
    env = FrozenLakeEnv(
        map_data=MAPS[args.map],
        params={'slippery': bool(args.slippery)}
    )
    agent = QLearningAgent(env, QL_PARAMS)

    # Entrenamiento
    print(f"Entrenando con mapa {args.map} ({'resbaladizo' if args.slippery else 'no resbaladizo'})...")
    stats = agent.train(episodes=args.episodes)

    # Guardar modelo
    model_path = f"output/models/qtable_{args.map}_{'slippery' if args.slippery else 'no_slippery'}.pkl"
    agent.save_model(model_path)
    print(f"Modelo guardado en {model_path}")

if __name__ == "__main__":
    main()