# Прогнозен изкуствен интелект (Predictive AI)
## Предсказване на цени на апартаменти в София с FastAI

### Dataset

Използваме **Bulgaria Real Estate Listings** от Kaggle (https://www.kaggle.com/datasets/gabrielagencheva/bulgaria-real-estate-listings). Dataset-ът съдържа реални обяви за имоти от imot.bg. За целите на проекта филтрираме само **обяви за продажба на апартаменти в София** — около **22 000 записа** с характеристики като площ (m2), тип апартамент (1-стаен, 2-стаен, мезонет и др.), тип строителство (тухла, панел и др.), етаж, брой етажи, година на строеж, квартал, цена на м² и 46 вида удобства (асансьор, паркинг, климатик, обзавеждане и др.). Целевата променлива е **цената на апартамента в EUR**.

### Модел

Тренираме **FastAI Tabular Learner** — напълно свързана невронна мрежа, специализирана за таблични данни. Моделът включва:

- **Embedding слоеве** за категорийни променливи (квартал, тип апартамент, тип строителство)
- **Два скрити слоя** с 400 и 200 неврона
- **Batch Normalization** и **Dropout 30%** за регуларизация
- **Weight decay 0.1** за допълнителна регуларизация
- **Log трансформация** на целевата променлива за по-нормално разпределение
- **OneCycleLR** learning rate scheduler за ефективно обучение
- Preprocessing: `Categorify`, `FillMissing` (медиана), `Normalize`

Обучението е 15 епохи с learning rate 0.001.

### Резултат

Моделът предсказва продажната цена на апартамент в София по неговите характеристики. Резултатите се оценяват на валидационно множество (20% от данните) чрез:

| Метрика | Описание |
|---------|----------|
| **RMSE** | Root Mean Squared Error — средна квадратична грешка в EUR |
| **MAE** | Mean Absolute Error — средна абсолютна грешка в EUR |
| **R² Score** | Коефициент на детерминация (1.0 = перфектно) |

Най-важните характеристики за предсказването са: площ (area_m2), цена на м² (price_per_m2), квартал (neighborhood), тип строителство (construction_type).

### Анализ на загубите (Validation Loss)

Проектът включва детайлен анализ на training и validation loss по време на обучението:

- **Validation Loss по епохи** — проследява как се променя грешката върху валидационното множество
- **Training vs Validation Loss** — сравнителна графика за диагностика на overfitting/underfitting
- **Числов отчет** — начална, крайна и минимална validation loss

### Инсталиране и стартиране

```bash
pip install fastai kaggle pandas matplotlib scikit-learn

# Изтегляне на данните от Kaggle
kaggle datasets download -d gabrielagencheva/bulgaria-real-estate-listings -p data_sofia/ --unzip

# Стартиране на notebook
jupyter notebook apartment_prices_prediction.ipynb

# Или стартиране на Python скрипт
python apartment_prices_prediction.py
```

### Технологии
- Python 3.x
- FastAI 2.x / PyTorch
- pandas, numpy, matplotlib, scikit-learn
- Kaggle API
