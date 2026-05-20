import pandas as pd
from ..utils.logger import get_logger

logger = get_logger(__name__)

def prepare_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Полная предобработка сырого датасета (из EDA.ipynb).
    """
    logger.debug("Начало предобработки данных")
    original_len = len(df)

    df = df.drop_duplicates()
    df = df.drop(['ID'], axis=1)

    df['Price'] = df['Price'] * 73

    df['Manufacturer'] = df['Manufacturer'].str.lower()
    manufacturer_counts = df['Manufacturer'].value_counts()
    rare_manufacturers = manufacturer_counts[manufacturer_counts < 15].index
    df['Manufacturer'] = df['Manufacturer'].replace(rare_manufacturers, 'other')

    df['Model'] = df['Model'].str.lower()
    model_counts = df['Model'].value_counts()
    rare_models = model_counts[model_counts < 10].index
    df['Model'] = df['Model'].replace(rare_models, 'other')

    df.loc[df['Fuel type'] == 'Plug-in Hybrid', 'Fuel type'] = 'hybrid'
    df.loc[(df['Fuel type'] == 'CNG') | (df['Fuel type'] == 'LPG'), 'Fuel type'] = 'gas'
    df['Fuel type'] = df['Fuel type'].str.lower()

    # Приводим к строке, если нужно
    if not pd.api.types.is_string_dtype(df['Engine volume']):
        df['Engine volume'] = df['Engine volume'].astype(str)
    df['Engine volume'] = df['Engine volume'].str.replace(" Turbo", "")
    df['Engine volume'] = df['Engine volume'].astype(float)

    if not pd.api.types.is_string_dtype(df['Mileage']):
        df['Mileage'] = df['Mileage'].astype(str)
    df['Mileage'] = df['Mileage'].str.replace(' km', '').astype(int)

    df['Drive wheels'] = df['Drive wheels'].str.lower()

    df.loc[df['Doors'] == '04-May', 'Doors'] = '4'
    df.loc[df['Doors'] == '02-Mar', 'Doors'] = '2'

    df.loc[df['Wheel'] == 'Left wheel', 'Wheel'] = 'left'
    df.loc[df['Wheel'] == 'Right-hand drive', 'Wheel'] = 'right'

    df['Color'] = df['Color'].str.lower()

    # Фильтрация выбросов
    df = df[(df['Price'] > 50000) & (df['Price'] < df['Price'].quantile(0.98))]
    df = df[(df['Prod. year'] >= 1970) & (df['Prod. year'] <= 2026)]
    df = df[df['Mileage'] >= 0]
    df = df[df['Engine volume'] > 0 & (df['Engine volume'] < df['Engine volume'].quantile(0.98))]
    df = df[(df['Cylinders'] > 0) & (df['Cylinders'] < 16)]
    df = df[df['Mileage'] < 1_000_000]

    logger.debug(f"Удалено {original_len - len(df)} строк при предобработке")
    logger.info(f"Предобработка завершена. Итоговый размер: {df.shape}")
    return df