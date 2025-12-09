Добавлен новый эндпоинт `quality-flags-from-csv`

Эндпоинт принимает CSV-файл, запускает EDA-ядро (summarize_dataset + missing_table + compute_quality_flags) и возвращает флаги оценки качества данных.

Пример вызова:
```
http://127.0.0.1:8000/quality-flags-from-csv
```
Пример ответа:
```
{
  "flags": {
    "too_few_rows": "True",
    "too_many_columns": "False",
    "max_missing_share": "0.05555555555555555",
    "too_many_missing": "False",
    "has_constant_columns": "False",
    "has_suspicious_id_duplicates": "True",
    "quality_score": "0.7444444444444445"
  }
}
```


