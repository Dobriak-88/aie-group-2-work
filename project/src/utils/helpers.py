import os
from pathlib import Path

# Корень исходного кода (папка src)
SRC_ROOT = Path(__file__).parent.parent
# Корень проекта (папка, содержащая src)
PROJECT_ROOT = SRC_ROOT.parent

# Чтение переменных окружения с fallback-значениями (пути внутри src)
DATA_PATH = Path(os.getenv("DATA_PATH", SRC_ROOT / "data" / "cars.csv"))
ARTIFACTS_PATH = Path(os.getenv("ARTIFACTS_PATH", SRC_ROOT / "artifacts"))
LOGS_PATH = Path(os.getenv("LOGS_PATH", SRC_ROOT / "logs"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# Создаём папки, если их нет
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
LOGS_PATH.mkdir(parents=True, exist_ok=True)