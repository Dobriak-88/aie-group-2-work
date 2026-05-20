import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ..data.loader import load_data
from ..features.build_features import build_preprocessor, NUM_FEATURES, CAT_FEATURES
from .utils import save_model, save_preprocessor, save_json, compute_metrics
from ..utils.helpers import ARTIFACTS_PATH, RANDOM_STATE
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Лучшие параметры, полученные из экспериментов в ноутбуке best_model_tune.ipynb
# (GridSearchCV дал эти значения)
BEST_PARAMS = {
    "max_depth": None,
    "max_features": 0.5,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "n_estimators": 300,
}

def train_pipeline():
    """Обучает модель с лучшими параметрами и сохраняет артефакты."""
    logger.info("Запуск пайплайна обучения")
    df = load_data()
    X = df[NUM_FEATURES + CAT_FEATURES]
    y = np.log(df['Price'])

    # Сохраняем списки допустимых производителей и моделей (после предобработки)
    valid_manufacturers = sorted(df['Manufacturer'].unique())
    valid_models = sorted(df['Model'].unique())
    ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)
    save_json(valid_manufacturers, ARTIFACTS_PATH / "valid_manufacturers.json")
    save_json(valid_models, ARTIFACTS_PATH / "valid_models.json")
    logger.info(f"Сохранено {len(valid_manufacturers)} производителей и {len(valid_models)} моделей")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    logger.info(f"Обучающая выборка: {X_train.shape}, тестовая: {X_test.shape}")

    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)
    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    logger.debug("Данные преобразованы")

    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **BEST_PARAMS)
    model.fit(X_train_processed, y_train)
    logger.info("Модель RandomForest обучена")

    y_pred_log = model.predict(X_test_processed)
    y_pred = np.exp(y_pred_log)
    y_true = np.exp(y_test)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Метрики на тесте: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}")

    # Сохранение артефактов
    save_model(model, ARTIFACTS_PATH / "random_forest_best.joblib")
    save_preprocessor(preprocessor, ARTIFACTS_PATH / "preprocessor.joblib")
    save_json(metrics, ARTIFACTS_PATH / "final_metrics.json")
    logger.info(f"Артефакты сохранены в {ARTIFACTS_PATH}")

    return model, preprocessor, metrics