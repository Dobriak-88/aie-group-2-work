import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Списки признаков
NUM_FEATURES = ['Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags']
CAT_FEATURES = [
    'Manufacturer', 'Model', 'Category', 'Leather interior', 'Fuel type',
    'Gear box type', 'Drive wheels', 'Doors', 'Wheel', 'Color'
]

def build_preprocessor() -> ColumnTransformer:
    """Создаёт ColumnTransformer для масштабирования и one-hot кодирования."""
    logger.info("Создание ColumnTransformer")
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUM_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES)
        ]
    )

def get_feature_names(preprocessor: ColumnTransformer) -> np.ndarray:
    """Возвращает имена всех признаков после трансформации."""
    cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CAT_FEATURES)
    return np.concatenate([NUM_FEATURES, cat_names])