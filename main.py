import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1. CARGA DE DATOS
# =========================

airbnb_data = pd.read_csv("./airbnb-listings-extract.csv", sep=";", decimal='.')

# =========================
# 2. LIMPIEZA DE PRICE
# =========================

airbnb_data['Price'] = (
    airbnb_data['Price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
)

airbnb_data['Price'] = pd.to_numeric(airbnb_data['Price'], errors='coerce')
airbnb_data = airbnb_data.dropna(subset=['Price'])

# =========================
# 3. LIMPIEZA DE %
# =========================

for col in ['Host Response Rate', 'Host Acceptance Rate']:
    airbnb_data[col] = (
        airbnb_data[col]
        .astype(str)
        .str.replace('%','')
    )
    airbnb_data[col] = pd.to_numeric(airbnb_data[col], errors='coerce')

# =========================
# 4. FEATURES
# =========================

columns = [
    'Accommodates',
    'Bathrooms',
    'Bedrooms',
    'Square Feet',
    'Room Type',
    'Latitude',
    'Longitude',
    'Review Scores Rating',
    'Number of Reviews',
    'Reviews per Month',
    'Availability 30',
    'Availability 90',
    'Availability 365',
    'Host Listings Count',
    'Host Total Listings Count',
]

X = airbnb_data[columns]
y = airbnb_data['Price']

# =========================
# 5. LOG TRANSFORM
# =========================

y = np.log1p(y)

# =========================
# 6. TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 7. ELIMINACIÓN DE OUTLIERS 
# =========================

Q1 = y_train.quantile(0.25)
Q3 = y_train.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

mask = (y_train >= lower) & (y_train <= upper)

X_train = X_train[mask]
y_train = y_train[mask]

# =========================
# 8. ONE HOT ENCODING
# =========================

X_train = pd.get_dummies(X_train, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Alinear columnas
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# =========================
# 9. IMPUTACIÓN
# =========================

imputer = SimpleImputer(strategy="median")

X_train = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)

X_test = pd.DataFrame(
    imputer.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# =========================
# 10. MODELO + GRID SEARCH
# =========================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=3,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

model = grid.best_estimator_

print("Mejores parámetros:", grid.best_params_)

# =========================
# 11. PREDICCIÓN
# =========================

pred = model.predict(X_test)

pred_real = np.expm1(pred)
y_test_real = np.expm1(y_test)

# =========================
# 12. EVALUACIÓN
# =========================

mae = mean_absolute_error(y_test_real, pred_real)
r2 = r2_score(y_test_real, pred_real)

print("MAE:", mae)
print("R2:", r2)

# =========================
# 13. CROSS VALIDATION
# =========================

scores = cross_val_score(
    model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_absolute_error"
)

print("MAE medio CV:", -scores.mean())

# =========================
# 14. GRÁFICO REAL VS PREDICHO
# =========================

plt.scatter(y_test_real, pred_real)

plt.plot(
    [y_test_real.min(), y_test_real.max()],
    [y_test_real.min(), y_test_real.max()]
)

plt.xlabel("Precio real")
plt.ylabel("Precio predicho")
plt.title("Real vs Predicho")

plt.show()

# =========================
# 15. RESIDUOS
# =========================

residuals = y_test_real - pred_real

plt.scatter(pred_real, residuals)
plt.axhline(0)

plt.xlabel("Precio predicho")
plt.ylabel("Error")
plt.title("Residuos")

plt.show()

# =========================
# 16. IMPORTANCIA VARIABLES
# =========================

importances = model.feature_importances_

feature_importance = pd.Series(
    importances,
    index=X_train.columns
)

feature_importance.sort_values().tail(15).plot.barh()

plt.title("Variables más importantes")

plt.show()
