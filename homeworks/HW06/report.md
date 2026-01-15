# HW06 – Report

> Файл: `homeworks/HW06/report.md`  
> Важно: не меняйте названия разделов (заголовков). Заполняйте текстом и/или вставляйте результаты.

## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-02.csv`
- Размер: (18000, 39)
- Целевая переменная: `target`, классы: 1 и 0, рапределение: 26% и 74% от общего количества
- Признаки: числовые

## 2. Protocol

- Разбиение: 0.25
- Подбор: CV на train (сколько фолдов, что оптимизировали)
- Метрики: accuracy, F1, ROC-AUC (и почему эти метрики уместны именно здесь)

## 3. Models

Опишите, какие модели сравнивали и какие гиперпараметры подбирали.

Минимум:

- DummyClassifier (baseline)
- LogisticRegression (baseline из S05)
- DecisionTreeClassifier (контроль сложности: `max_depth` + `min_samples_leaf` или `ccp_alpha`)
- RandomForestClassifier
- Один boosting (AdaBoost / GradientBoosting / HistGradientBoosting)

Опционально:

- StackingClassifier (с CV-логикой)

## 4. Results

- Таблица/список финальных метрик на test по всем моделям
- Победитель (по ROC-AUC или по согласованному критерию) и краткое объяснение

## 5. Analysis

- Устойчивость: что будет, если поменять `random_state` (хотя бы 5 прогонов для 1-2 моделей) – кратко
- Ошибки: confusion matrix для лучшей модели + комментарий
- Интерпретация: permutation importance (top-10/15) + выводы

## 6. Conclusion

3-6 коротких тезисов: что вы поняли про деревья/ансамбли и про честный ML-протокол.