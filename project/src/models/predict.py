import numpy as np
import pandas as pd
from .utils import load_model, load_preprocessor, load_json
from ..features.build_features import NUM_FEATURES, CAT_FEATURES
from ..utils.helpers import ARTIFACTS_PATH
from ..utils.logger import get_logger

logger = get_logger(__name__)

def load_pipeline():
    """Загружает модель, препроцессор и справочники."""
    logger.info("Загрузка модели и препроцессора")
    model = load_model(ARTIFACTS_PATH / "random_forest_best.joblib")
    preprocessor = load_preprocessor(ARTIFACTS_PATH / "preprocessor.joblib")
    valid_manufacturers = set(load_json(ARTIFACTS_PATH / "valid_manufacturers.json"))
    valid_models = set(load_json(ARTIFACTS_PATH / "valid_models.json"))
    logger.debug(f"Загружено {len(valid_manufacturers)} производителей, {len(valid_models)} моделей")
    return model, preprocessor, valid_manufacturers, valid_models

def normalize_new_data(df: pd.DataFrame, valid_manufacturers: set, valid_models: set) -> pd.DataFrame:
    """
    Приводит производителя и модель к нижнему регистру.
    Заменяет на 'other', если значения нет в списках.
    Все остальные категориальные колонки заполняет пустой строкой вместо None.
    """
    df = df.copy()
    
    # 1. Обработка производителя и модели
    df['Manufacturer'] = df['Manufacturer'].astype(str).str.lower().str.strip()
    df['Model'] = df['Model'].astype(str).str.lower().str.strip()
    df['Manufacturer'] = df['Manufacturer'].apply(lambda x: x if x in valid_manufacturers else 'other')
    df['Model'] = df['Model'].apply(lambda x: x if x in valid_models else 'other')
    
    # 2. Для всех категориальных признаков (включая Color, Doors и т.д.) заменяем None/NaN на пустую строку
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
    
    logger.debug(f"После нормализации: производителей unique={df['Manufacturer'].nunique()}, "
                 f"моделей unique={df['Model'].nunique()}")
    return df

def predict_price(model, preprocessor, X: pd.DataFrame, valid_manufacturers: set, valid_models: set) -> np.ndarray:
    """
    Предсказывает цены в рублях.
    Предварительно нормализует производителя и модель.
    """
    logger.debug(f"Предсказание для {X.shape[0]} объектов")
    X_clean = normalize_new_data(X, valid_manufacturers, valid_models)
    X_proc = preprocessor.transform(X_clean[NUM_FEATURES + CAT_FEATURES])
    log_price = model.predict(X_proc)
    prices = np.exp(log_price)
    logger.debug(f"Диапазон предсказанных цен: [{prices.min():.2f}, {prices.max():.2f}]")
    return prices