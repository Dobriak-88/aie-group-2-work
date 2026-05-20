from __future__ import annotations

import typer
import pandas as pd
from pathlib import Path
from ..models.train import train_pipeline
from ..models.predict import load_pipeline, predict_price
from ..models.utils import compute_metrics
from ..features.build_features import NUM_FEATURES, CAT_FEATURES
from ..utils.helpers import ARTIFACTS_PATH
from ..utils.logger import get_logger

logger = get_logger(__name__)
app = typer.Typer(help="Car Price Prediction CLI")

@app.command()
def train(force: bool = False):
    """Обучить модель и сохранить артефакты."""
    logger.info("Выполнение команды train")
    model_path = ARTIFACTS_PATH / "random_forest_best.joblib"
    preprocessor_path = ARTIFACTS_PATH / "preprocessor.joblib"
    if not force and model_path.exists() and preprocessor_path.exists():
        logger.info("Артефакты уже существуют. Используйте --force для переобучения.")
        return
    train_pipeline()
    logger.info("Обучение завершено")

@app.command()
def predict(
    input_csv: Path = typer.Argument(..., help="Путь к CSV с признаками"),
    output_csv: Path = typer.Option(None, help="Путь для сохранения предсказаний"),
):
    """Сделать предсказание для нового CSV."""
    logger.info(f"Запуск предсказания для {input_csv}")
    model, preprocessor, valid_manuf, valid_models = load_pipeline()
    df = pd.read_csv(input_csv)
    missing = [c for c in NUM_FEATURES + CAT_FEATURES if c not in df.columns]
    if missing:
        logger.error(f"Отсутствуют колонки: {missing}")
        raise typer.Exit(code=1)
    prices = predict_price(model, preprocessor, df, valid_manuf, valid_models)
    if output_csv:
        df["Predicted_Price"] = prices
        df.to_csv(output_csv, index=False)
        logger.info(f"Предсказания сохранены в {output_csv}")
    else:
        for i, p in enumerate(prices[:10]):
            print(f"{i+1}: {p:.2f} руб.")
        if len(prices) > 10:
            print("... (показаны первые 10)")

@app.command()
def evaluate(
    test_csv: Path = typer.Option(None, help="CSV с тестовыми данными (должна быть колонка Price)"),
):
    """Оценить модель на тестовых данных."""
    logger.info("Запуск оценки модели")
    model, preprocessor, valid_manuf, valid_models = load_pipeline()
    if test_csv:
        df = pd.read_csv(test_csv)
        if "Price" not in df.columns:
            logger.error("В CSV отсутствует колонка 'Price'.")
            raise typer.Exit(1)
        X = df[NUM_FEATURES + CAT_FEATURES]
        y_true = df["Price"].values
    else:
        from ..data.loader import load_data
        from sklearn.model_selection import train_test_split
        import numpy as np
        df = load_data()
        X = df[NUM_FEATURES + CAT_FEATURES]
        y = np.log(df["Price"])
        _, X, _, y_true_log = train_test_split(X, y, test_size=0.2, random_state=42)
        y_true = np.exp(y_true_log)
    y_pred = predict_price(model, preprocessor, X, valid_manuf, valid_models)
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"RMSE = {metrics['rmse']:.4f}, MAE = {metrics['mae']:.4f}")
    print(f"RMSE = {metrics['rmse']:.4f}")
    print(f"MAE  = {metrics['mae']:.4f}")

if __name__ == "__main__":
    app()