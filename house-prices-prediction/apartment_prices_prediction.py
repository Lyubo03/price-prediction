"""
Прогнозен изкуствен интелект (Predictive AI)
Предсказване на цени на апартаменти в София с FastAI

Dataset: Bulgaria Real Estate Listings (Kaggle)
Модел: FastAI Tabular Learner
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from fastai.tabular.all import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# 1. Zarezhdane na dannite
# ============================================================

data_path = Path('data_sofia')
clean_path = data_path / 'clean'

if not clean_path.exists():
    import subprocess
    data_path.mkdir(exist_ok=True)
    subprocess.run([
        'kaggle', 'datasets', 'download',
        '-d', 'gabrielagencheva/bulgaria-real-estate-listings',
        '-p', 'data_sofia/', '--unzip'
    ])
    print('Dannite sa iztegleni ot Kaggle!')

listings = pd.read_csv(clean_path / 'listings.csv')
properties = pd.read_csv(clean_path / 'properties.csv')
geographies = pd.read_csv(clean_path / 'geographies.csv')
property_types = pd.read_csv(clean_path / 'property_types.csv')
construction_types = pd.read_csv(clean_path / 'construction_types.csv')
features = pd.read_csv(clean_path / 'features.csv')
property_features = pd.read_csv(clean_path / 'property_features.csv')

print(f'Obqvi: {len(listings)}, Imoti: {len(properties)}')

# ============================================================
# 2. Filtrirane za apartamenti v Sofia
# ============================================================

sofia_region = geographies[(geographies['name_bg'] == '\u0421\u043e\u0444\u0438\u044f') & (geographies['level'] == 'region')]
sofia_region_id = sofia_region['geo_id'].values[0]

sofia_ids = set([sofia_region_id])
localities = geographies[geographies['parent_id'] == float(sofia_region_id)]
sofia_ids.update(localities['geo_id'].tolist())
for lid in localities['geo_id'].tolist():
    areas = geographies[geographies['parent_id'] == float(lid)]
    sofia_ids.update(areas['geo_id'].tolist())

print(f'Geografski zoni v Sofia: {len(sofia_ids)}')

sofia_props = properties[properties['geo_id'].isin(sofia_ids)].copy()

df = sofia_props.merge(listings[['property_id', 'price', 'transaction_type']], on='property_id')
df = df[(df['transaction_type'] == 'sale') & (df['price'].notna()) & (df['price'] > 0)].copy()

df = df.merge(property_types[['property_type_id', 'name_en', 'category']], on='property_type_id', how='left')
df.rename(columns={'name_en': 'property_type', 'category': 'property_category'}, inplace=True)

# Filtrirane SAMO na apartamenti (apartment, multi-room apartment, maisonette, studio/attic)
apartment_types = ['apartment', 'multi-room apartment', 'maisonette', 'studio/attic']
df = df[df['property_type'].isin(apartment_types)].copy()
print(f'Apartamenti v Sofia: {len(df)}')

df = df.merge(construction_types[['construction_type_id', 'name_en']], on='construction_type_id', how='left')
df.rename(columns={'name_en': 'construction_type'}, inplace=True)

df = df.merge(geographies[['geo_id', 'name_bg']], on='geo_id', how='left')
df.rename(columns={'name_bg': 'neighborhood'}, inplace=True)

# Pivot na udobstvata
pf_sofia = property_features[property_features['property_id'].isin(df['property_id'])]
pf_with_names = pf_sofia.merge(features[['feature_id', 'name_en']], on='feature_id')
feat_pivot = pf_with_names.pivot_table(index='property_id', columns='name_en', aggfunc='size', fill_value=0).clip(upper=1)
df = df.merge(feat_pivot, on='property_id', how='left')
df[feat_pivot.columns] = df[feat_pivot.columns].fillna(0).astype(int)

# Dobavqne na cena na kvadraten metar
df['area_m2'] = pd.to_numeric(df['area_m2'], errors='coerce')
df['price_per_m2'] = df['price'] / df['area_m2']

# Premahvane na nenujni koloni i outliers
drop_cols = ['property_id', 'geo_id', 'property_type_id', 'construction_type_id', 'transaction_type']
df.drop(columns=drop_cols, inplace=True)

before = len(df)
df = df[(df['price'] >= 20000) & (df['price'] <= 1_000_000)].copy()
# Premahvame i nerealichni ceni na m2 (pod 200 ili nad 10000 EUR/m2)
df = df[(df['price_per_m2'] >= 200) & (df['price_per_m2'] <= 10000)].copy()
print(f'Premahanti outliers: {before - len(df)}')
print(f'Finalen dataset: {df.shape[0]} reda, {df.shape[1]} koloni')

# ============================================================
# 3. EDA
# ============================================================

print(f'\nStatistika na cenite (EUR):')
print(df['price'].describe())
print(f'\nStatistika na cenite na m2 (EUR/m2):')
print(df['price_per_m2'].describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['price'], bins=50, color='steelblue', edgecolor='black')
axes[0].set_title('Razpredelenie na cenite na apartamenti')
axes[0].set_xlabel('Cena (EUR)')
axes[1].hist(np.log1p(df['price']), bins=50, color='coral', edgecolor='black')
axes[1].set_title('Razpredelenie na log(Cena)')
axes[1].set_xlabel('log(Cena)')
plt.tight_layout()
plt.savefig('distribution.png', dpi=150)
print('Saved distribution.png')

# ============================================================
# 4. Podgotovka na dannite
# ============================================================

dep_var = 'price'

# Zapulvane na lipsvashti stojnosti
for col in ['area_m2', 'floor', 'total_floors', 'year_built', 'bedrooms', 'price_per_m2']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median())

for col in ['property_type', 'property_category', 'construction_type',
            'construction_status', 'neighborhood', 'gas', 'tec']:
    if col in df.columns:
        df[col] = df[col].fillna('unknown')

df['price'] = np.log1p(df['price'])

cat_names = ['property_type', 'construction_type',
             'construction_status', 'neighborhood', 'gas', 'tec']
cat_names = [c for c in cat_names if c in df.columns]

cont_names = ['area_m2', 'floor', 'total_floors', 'year_built', 'bedrooms', 'price_per_m2']
feature_cols = [c for c in df.columns if c not in cat_names + cont_names + [dep_var, 'property_category']]
feature_cols = [c for c in feature_cols if df[c].dtype in ['int64', 'float64', 'int32']]
cont_names = cont_names + feature_cols
cont_names = [c for c in cont_names if c in df.columns]

print(f'\nKategorijni koloni: {len(cat_names)}')
print(f'Chislovi koloni: {len(cont_names)}')

procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter(valid_pct=0.2, seed=42)(range_of(df))

to = TabularPandas(df, procs=procs, cat_names=cat_names, cont_names=cont_names,
                   y_names=dep_var, splits=splits, y_block=RegressionBlock())

print(f'Training: {len(to.train)}, Validation: {len(to.valid)}')

# ============================================================
# 5. Obuchenie
# ============================================================

dls = to.dataloaders(bs=64)

learn = tabular_learner(
    dls, layers=[400, 200], metrics=[rmse, mae],
    config=tabular_config(ps=[0.3, 0.3], embed_p=0.1),
    y_range=(df['price'].min() * 0.9, df['price'].max() * 1.1),
    wd=0.1
)

print('\nModel architecture:')
print(learn.model)

print('\nTraining (15 epochs)...')
learn.fit_one_cycle(15, lr_max=1e-3)

# ============================================================
# 6. Validation Loss analiz
# ============================================================

train_losses = [x.item() for x in learn.recorder.losses]
valid_losses = [x[0] if isinstance(x[0], float) else x[0].item() for x in learn.recorder.values]
n_epochs = len(valid_losses)

fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].plot(range(1, n_epochs + 1), valid_losses, 'o-', color='coral', label='Validation Loss', linewidth=2)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Validation Loss per epoch')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

batches_per_epoch = len(train_losses) // n_epochs
epoch_train_losses = []
for i in range(n_epochs):
    start = i * batches_per_epoch
    end = start + batches_per_epoch
    epoch_train_losses.append(np.mean(train_losses[start:end]))

axes[1].plot(range(1, n_epochs + 1), epoch_train_losses, 'o-', color='steelblue', label='Training Loss', linewidth=2, markersize=4)
axes[1].plot(range(1, n_epochs + 1), valid_losses, 's-', color='coral', label='Validation Loss', linewidth=2, markersize=4)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Training vs Validation Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('loss_analysis.png', dpi=150)
print('\nSaved loss_analysis.png')

print(f'\nStart Validation Loss (epoch 1): {valid_losses[0]:.6f}')
print(f'Final Validation Loss (epoch {n_epochs}): {valid_losses[-1]:.6f}')
print(f'Best Validation Loss: {min(valid_losses):.6f} (epoch {valid_losses.index(min(valid_losses)) + 1})')
print(f'Final Training Loss: {epoch_train_losses[-1]:.6f}')

# ============================================================
# 7. Ocenka
# ============================================================

preds, targets = learn.get_preds()

preds_real = np.expm1(preds.numpy().flatten())
targets_real = np.expm1(targets.numpy().flatten())

rmse_val = np.sqrt(mean_squared_error(targets_real, preds_real))
mae_val = mean_absolute_error(targets_real, preds_real)
r2_val = r2_score(targets_real, preds_real)

print('\n' + '=' * 50)
print('REZULTATI NA VALIDACIONNOTO MNOZHESTVO')
print('=' * 50)
print(f'RMSE: {rmse_val:,.2f} EUR')
print(f'MAE:  {mae_val:,.2f} EUR')
print(f'R2 Score: {r2_val:.4f}')
print(f'Sredna greshka: {(mae_val / targets_real.mean()) * 100:.1f}%')
print('=' * 50)

# Vizualizaciq
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].scatter(targets_real, preds_real, alpha=0.3, s=10, color='steelblue')
max_val = max(targets_real.max(), preds_real.max())
axes[0].plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
axes[0].set_xlabel('Real price (EUR)')
axes[0].set_ylabel('Predicted price (EUR)')
axes[0].set_title('Predicted vs Real apartment prices - Sofia')
axes[0].legend()

errors = preds_real - targets_real
axes[1].hist(errors, bins=50, color='coral', edgecolor='black')
axes[1].axvline(x=0, color='black', linestyle='--')
axes[1].set_xlabel('Error (EUR)')
axes[1].set_title('Error distribution')
plt.tight_layout()
plt.savefig('results.png', dpi=150)
print('Saved results.png')

# ============================================================
# 8. Zapazване
# ============================================================

learn.export('sofia_apartment_prices_model.pkl')
print('\nModel saved as sofia_apartment_prices_model.pkl')
