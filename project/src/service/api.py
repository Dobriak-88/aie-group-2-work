from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional
import pandas as pd
from ..models.predict import load_pipeline, predict_price
from ..features.build_features import NUM_FEATURES, CAT_FEATURES
from ..utils.logger import get_logger

logger = get_logger(__name__)

# Глобальные переменные для модели и справочников
model = None
preprocessor = None
valid_manufacturers = set()
valid_models = set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загружает модель и справочники при старте сервера."""
    global model, preprocessor, valid_manufacturers, valid_models
    logger.info("Загрузка модели и справочников")
    try:
        model, preprocessor, valid_manufacturers, valid_models = load_pipeline()
        logger.info(f"Загружено {len(valid_manufacturers)} производителей, {len(valid_models)} моделей")
    except Exception as e:
        logger.error(f"Ошибка загрузки: {e}")
    yield
    logger.info("API shutdown")


app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
    description="Сервис предсказания цены автомобиля на основе RandomForest",
    lifespan=lifespan,
)


class CarFeatures(BaseModel):
    # Числовые признаки (с валидацией через Field)
    prod_year: int = Field(..., alias="Prod. year", ge=1970, le=2026, description="Год выпуска (1970-2026)")
    engine_volume: float = Field(..., alias="Engine volume", gt=0, description="Объём двигателя в литрах")
    mileage: int = Field(..., alias="Mileage", ge=0, description="Пробег в километрах")
    cylinders: float = Field(..., alias="Cylinders", gt=0, le=16, description="Количество цилиндров")
    airbags: int = Field(..., alias="Airbags", ge=0, description="Количество подушек безопасности")

    # Категориальные признаки (обязательные, жёсткая валидация)
    manufacturer: Optional[str] = Field(None, alias="Manufacturer", description="Производитель (см. /manufacturers). Если None или не в списке → заменится на 'other'.")
    model_name: Optional[str] = Field(None, alias="Model", description="Модель (см. /models). Если None или не в списке → заменится на 'other'.")
    category: str = Field(..., alias="Category", description="Тип кузова. Допустимые значения: 'Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup'")
    leather_interior: str = Field(..., alias="Leather interior", description="Кожаный салон: 'Yes' или 'No'")
    fuel_type: str = Field(..., alias="Fuel type", description="Тип топлива: 'petrol', 'diesel', 'hybrid', 'gas'")
    gear_box_type: str = Field(..., alias="Gear box type", description="Коробка передач: 'Automatic', 'Tiptronic', 'Variator', 'Manual'")
    drive_wheels: str = Field(..., alias="Drive wheels", description="Привод: 'front', 'rear', '4x4'")
    doors: str = Field(..., alias="Doors", description="Количество дверей: '2', '4', '>5'")
    wheel: str = Field(..., alias="Wheel", description="Руль: 'left' или 'right'")
    color: Optional[str] = Field(None, alias="Color", description="Цвет (любая строка или None)")

    # Валидаторы
    @field_validator('category', mode='before')
    def validate_category(cls, v):
        allowed = {'Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup'}
        if v is None:
            raise ValueError("Category обязателен")
        v_str = str(v).strip()
        if v_str not in allowed:
            raise ValueError(f"Недопустимое значение Category. Допустимо: {sorted(allowed)}")
        return v_str

    @field_validator('leather_interior', mode='before')
    def validate_leather_interior(cls, v):
        if v is None:
            raise ValueError("Leather interior обязателен")
        v_str = str(v).strip().lower()
        if v_str not in ('yes', 'no'):
            raise ValueError("Leather interior должен быть 'Yes' или 'No'")
        return v_str.capitalize()  # Приводим к виду из обучения ('Yes'/'No')

    @field_validator('fuel_type', mode='before')
    def validate_fuel_type(cls, v):
        allowed = {'petrol', 'diesel', 'hybrid', 'gas'}
        if v is None:
            raise ValueError("Fuel type обязателен")
        v_lower = str(v).strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Недопустимое значение Fuel type. Допустимо: {sorted(allowed)}")
        return v_lower

    @field_validator('gear_box_type', mode='before')
    def validate_gear_box_type(cls, v):
        allowed = {'automatic', 'tiptronic', 'variator', 'manual'}
        if v is None:
            raise ValueError("Gear box type обязателен")
        v_lower = str(v).strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Недопустимое значение Gear box type. Допустимо: {sorted(allowed)}")
        # Приводим к виду из обучения (первая буква заглавная)
        return v_lower.capitalize()

    @field_validator('drive_wheels', mode='before')
    def validate_drive_wheels(cls, v):
        allowed = {'front', 'rear', '4x4'}
        if v is None:
            raise ValueError("Drive wheels обязателен")
        v_lower = str(v).strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Недопустимое значение Drive wheels. Допустимо: {sorted(allowed)}")
        return v_lower

    @field_validator('doors', mode='before')
    def validate_doors(cls, v):
        allowed = {'2', '4', '>5'}
        if v is None:
            raise ValueError("Doors обязателен")
        v_str = str(v).strip()
        if v_str not in allowed:
            raise ValueError(f"Недопустимое значение Doors. Допустимо: {sorted(allowed)}")
        return v_str

    @field_validator('wheel', mode='before')
    def validate_wheel(cls, v):
        allowed = {'left', 'right'}
        if v is None:
            raise ValueError("Wheel обязателен")
        v_lower = str(v).strip().lower()
        if v_lower not in allowed:
            raise ValueError(f"Недопустимое значение Wheel. Допустимо: {sorted(allowed)}")
        return v_lower

    # Для цвета валидация не нужна – любая строка или None
    # Производитель и модель валидируются в normalize_car (там заменяются на 'other' при None или неизвестном)

    class Config:
        validate_by_name = True
        json_schema_extra = {
            "example": {
                "Prod. year": 2015,
                "Engine volume": 2.0,
                "Mileage": 50000,
                "Cylinders": 4,
                "Airbags": 4,
                "Manufacturer": "toyota",
                "Model": "camry",
                "Category": "Sedan",
                "Leather interior": "No",
                "Fuel type": "petrol",
                "Gear box type": "Automatic",
                "Drive wheels": "front",
                "Doors": "4",
                "Wheel": "left",
                "Color": "silver"
            }
        }


def normalize_car(car: CarFeatures) -> dict:
    """Преобразует pydantic-модель в словарь, заменяя manufacturer/model на 'other' при необходимости."""
    data = car.dict(by_alias=True)

    # Обработка производителя
    manuf = data.get('Manufacturer')
    if manuf is None or not isinstance(manuf, str):
        data['Manufacturer'] = 'other'
    else:
        manuf_clean = manuf.strip().lower()
        data['Manufacturer'] = manuf_clean if manuf_clean in valid_manufacturers else 'other'

    # Обработка модели
    mdl = data.get('Model')
    if mdl is None or not isinstance(mdl, str):
        data['Model'] = 'other'
    else:
        model_clean = mdl.strip().lower()
        data['Model'] = model_clean if model_clean in valid_models else 'other'

    # Остальные поля оставляем как есть (они уже валидированы и не None)
    return data


@app.get("/health")
def health():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "model_loaded": model is not None}


@app.get("/manufacturers")
def get_manufacturers():
    """Возвращает список допустимых производителей (без 'other')."""
    return {"manufacturers": sorted([m for m in valid_manufacturers if m != 'other'])}


@app.get("/models")
def get_models():
    """Возвращает список допустимых моделей (без 'other')."""
    return {"models": sorted([m for m in valid_models if m != 'other'])}


@app.post("/predict")
def predict(car: CarFeatures):
    """Предсказать цену автомобиля."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Модель не загружена")

    logger.info(f"Запрос для {car.manufacturer} {car.model_name}")
    data = normalize_car(car)
    X = pd.DataFrame([data])[NUM_FEATURES + CAT_FEATURES]

    try:
        price = predict_price(model, preprocessor, X, valid_manufacturers, valid_models)[0]
        logger.info(f"Предсказанная цена: {price:.2f} руб.")
        return {"predicted_price": float(price)}
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))