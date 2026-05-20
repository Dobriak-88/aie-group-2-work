# Тестирование функций train_pipeline, load_pipeline, predict_price, compute_metrics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np


@pytest.fixture
def temp_data_dir(tmp_path):
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    data_dir.mkdir()
    artifacts_dir.mkdir()

    test_csv = data_dir / "cars.csv"
    sample_data = {
        "ID": [1, 2, 3],
        "Price": [1000000, 1500000, 800000],
        "Manufacturer": ["toyota", "honda", "bmw"],
        "Model": ["camry", "civic", "x5"],
        "Prod. year": [2015, 2018, 2010],
        "Category": ["Sedan", "Hatchback", "Jeep"],
        "Leather interior": ["No", "Yes", "Yes"],
        "Fuel type": ["Petrol", "Petrol", "Diesel"],
        "Engine volume": ["2.0", "1.6", "3.0"],
        "Mileage": ["50000 km", "30000 km", "120000 km"],
        "Cylinders": [4, 4, 6],
        "Gear box type": ["Automatic", "Manual", "Automatic"],
        "Drive wheels": ["Front", "Front", "4x4"],
        "Doors": ["04-May", "04-May", "04-May"],
        "Wheel": ["Left wheel", "Left wheel", "Left wheel"],
        "Color": ["Silver", "Red", "Black"],
        "Airbags": [4, 2, 8],
    }
    df = pd.DataFrame(sample_data)
    df['Engine volume'] = df['Engine volume'].astype(str)   # фикс
    df.to_csv(test_csv, index=False)
    return tmp_path


def test_train_pipeline(temp_data_dir, monkeypatch):
    monkeypatch.setattr("src.utils.helpers.DATA_PATH", temp_data_dir / "data" / "cars.csv")
    monkeypatch.setattr("src.utils.helpers.ARTIFACTS_PATH", temp_data_dir / "artifacts")

    from src.models.train import train_pipeline

    model, preprocessor, metrics = train_pipeline()

    assert (temp_data_dir / "artifacts" / "random_forest_best.joblib").exists()
    assert (temp_data_dir / "artifacts" / "preprocessor.joblib").exists()
    assert "rmse" in metrics
    assert "mae" in metrics


def test_predict_pipeline(temp_data_dir, monkeypatch):
    monkeypatch.setattr("src.utils.helpers.DATA_PATH", temp_data_dir / "data" / "cars.csv")
    monkeypatch.setattr("src.utils.helpers.ARTIFACTS_PATH", temp_data_dir / "artifacts")

    from src.models.train import train_pipeline
    from src.models.predict import load_pipeline, predict_price

    train_pipeline()
    model, preprocessor, valid_manuf, valid_models = load_pipeline()

    new_data = pd.DataFrame({
        "Prod. year": [2015, 2020],
        "Engine volume": [2.0, 1.5],
        "Mileage": [50000, 10000],
        "Cylinders": [4, 4],
        "Airbags": [4, 4],
        "Manufacturer": ["toyota", "honda"],
        "Model": ["camry", "civic"],
        "Category": ["Sedan", "Hatchback"],
        "Leather interior": ["No", "Yes"],
        "Fuel type": ["petrol", "petrol"],
        "Gear box type": ["Automatic", "Manual"],
        "Drive wheels": ["front", "front"],
        "Doors": ["4", "4"],
        "Wheel": ["left", "left"],
        "Color": ["silver", "red"],
    })

    prices = predict_price(model, preprocessor, new_data, valid_manuf, valid_models)
    assert len(prices) == 2
    assert np.all(prices > 0)


def test_evaluate_on_test_set(temp_data_dir, monkeypatch):
    monkeypatch.setattr("src.utils.helpers.DATA_PATH", temp_data_dir / "data" / "cars.csv")
    monkeypatch.setattr("src.utils.helpers.ARTIFACTS_PATH", temp_data_dir / "artifacts")

    from src.models.train import train_pipeline
    from src.models.predict import load_pipeline, predict_price
    from src.models.utils import compute_metrics

    train_pipeline()
    model, preprocessor, valid_manuf, valid_models = load_pipeline()

    test_data = pd.DataFrame({
        "Prod. year": [2015, 2020],
        "Engine volume": [2.0, 1.5],
        "Mileage": [50000, 10000],
        "Cylinders": [4, 4],
        "Airbags": [4, 4],
        "Manufacturer": ["toyota", "honda"],
        "Model": ["camry", "civic"],
        "Category": ["Sedan", "Hatchback"],
        "Leather interior": ["No", "Yes"],
        "Fuel type": ["petrol", "petrol"],
        "Gear box type": ["Automatic", "Manual"],
        "Drive wheels": ["front", "front"],
        "Doors": ["4", "4"],
        "Wheel": ["left", "left"],
        "Color": ["silver", "red"],
        "Price": [1200000, 800000],
    })

    prices = predict_price(model, preprocessor, test_data, valid_manuf, valid_models)
    metrics = compute_metrics(test_data["Price"].values, prices)
    assert "rmse" in metrics
    assert metrics["rmse"] >= 0
    assert "mae" in metrics
    assert metrics["mae"] >= 0