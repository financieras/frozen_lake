# frozen_lake

## Estructura del Proyecto

```bash
frozen_lake/
│
├── .venv/ # Entorno virtual (ignorado por git)
├── config/
│   ├── init.py
│   ├── params.py # Hiperparámetros de Q-Learning
│   └── maps.py # Mapas predefinidos
│
├── src/
│   ├── init.py
│   ├── environment.py # Lógica del entorno del juego
│   ├── qlearning.py # Implementación de Q-Learning
│   ├── visualization.py # Visualización en terminal
│   └── train.py # Script de entrenamiento
│
├── tests/
│   ├── init.py
│   ├── test_environment.py
│   └── test_qlearning.py
│
├── output/
│   ├── models/ # Q-tables guardadas
│   └── plots/ # Gráficos de rendimiento
│
├── requirements.txt # Dependencias del proyecto
└── README.md # Este archivo
```


## Requisitos

- Python 3.8+
- pip

## Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/financieras/frozen_lake.git
cd frozen_lake
```

## Crear y activar entorno virtual
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# o .venv\Scripts\activate en Windows
```

## Instalar dependencias
```bash
pip install -r requirements.txt
```

# Uso
## Entrenamiento básico

```bash
python src/train.py --map 4x4 --episodes 10000
```

## Visualización de resultados
```bash
python src/visualization.py --model output/models/qtable_4x4.npy
```

## Parámetros disponibles

| Argumento       | Descripción                          | Valor por defecto |
|-----------------|-------------------------------------|-------------------|
| `--map`         | Tamaño del mapa (4x4 u 8x8)         | `4x4`             |
| `--episodes`    | Número de episodios de entrenamiento| `10000`           |
| `--slippery`    | Superficie resbaladiza (0/1)        | `1`               |
| `--render`      | Mostrar animación (0/1)             | `0`               |

## Configuración
Edita los archivos en config/ para ajustar:
- Hiperparámetros de Q-Learning (params.py)
- Diseño de mapas personalizados (maps.py)

## Ejecución de tests
```bash
pytest tests/ -v
```