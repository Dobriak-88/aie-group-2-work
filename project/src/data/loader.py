import pandas as pd
from .preprocess import prepare_raw_data
from ..utils.helpers import DATA_PATH
from ..utils.logger import get_logger

logger = get_logger(__name__)

def load_data() -> pd.DataFrame:
    """Загружает и предобрабатывает датасет."""
    logger.info(f"Загрузка данных из {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Загружено {df.shape[0]} строк, {df.shape[1]} столбцов")
    df = prepare_raw_data(df)
    return df