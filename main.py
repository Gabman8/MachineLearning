
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# =========================
# 1. CARGA DE DATOS
# =========================

airbnb_data = pd.read_csv("./airbnb-listings-extract.csv", sep=";", decimal='.')

# =========================
# 2. SELECCIÓN DE FEATURES
# =========================
# Se eligen variables relevantes para predecir el precio, evitando aquellas que no aportan valor predictivo
# y las que al ejecutar el modelo se demuestra que tampoco aportan valor
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

# =========================
# 3. LIMPIEZA DE LA VARIABLE OBJETIVO (PRICE)
# =========================

# eliminar símbolos de moneda
airbnb_data['Price'] = (
    airbnb_data['Price']
    .astype(str)
    .str.replace(r'[^\d.]', '', regex=True)
)

# convertir a número
airbnb_data['Price'] = pd.to_numeric(airbnb_data['Price'], errors='coerce')

# eliminar filas sin precio
airbnb_data = airbnb_data.dropna(subset=['Price'])

print("NaN en Price:", airbnb_data['Price'].isna().sum())

# =========================
# 4. LIMPIEZA DE COLUMNAS CON %
# =========================

for col in ['Host Response Rate', 'Host Acceptance Rate']:

    airbnb_data[col] = (
        airbnb_data[col]
        .astype(str)
        .str.replace('%','')
    )

    airbnb_data[col] = pd.to_numeric(airbnb_data[col], errors='coerce')

# =========================
# 5. TRAIN / TEST SPLIT
# =========================
X = airbnb_data[columns]
y = airbnb_data['Price']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# 6. ELIMINACIÓN DE OUTLIERS
# =========================

# eliminar precios extremos (mejora mucho el modelo)
y_train = y_train[y_train < 500]


# =========================
# 7. LOG TRANSFORM DEL TARGET
# =========================

# mejora el aprendizaje en distribuciones sesgadas
y = np.log1p(y)

# =========================
# 8. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# =========================

X = pd.get_dummies(X, drop_first=True)

# =========================
# 9. IMPUTACIÓN DE VALORES FALTANTES
# =========================

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

# =========================
# 10. NORMALIZACIÓN
# =========================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 11. OPTIMIZACIÓN DEL MODELO
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
# 12. PREDICCIÓN
# =========================

pred = model.predict(X_test)

# revertir log transform
pred_real = np.expm1(pred)
y_test_real = np.expm1(y_test)

# =========================
# 13. EVALUACIÓN
# =========================

mae = mean_absolute_error(y_test_real, pred_real)
r2 = r2_score(y_test_real, pred_real)

print("MAE:", mae)
print("R2:", r2)

# =========================
# 14. CROSS VALIDATION
# =========================

scores = cross_val_score(

    model,
    X,
    y,
    cv=5,
    scoring="neg_mean_absolute_error"

)

print("MAE medio CV:", -scores.mean())

# =========================
# 15. GRÁFICO REAL VS PREDICHO
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
# 16. GRÁFICO DE RESIDUOS
# =========================

residuals = y_test_real - pred_real

plt.scatter(pred_real, residuals)

plt.axhline(0)

plt.xlabel("Precio predicho")
plt.ylabel("Error")

plt.title("Residuos del modelo")

plt.show()

# =========================
# 17. IMPORTANCIA DE VARIABLES
# =========================

importances = model.feature_importances_

feature_importance = pd.Series(
    importances,
    index=X.columns
)

feature_importance.sort_values().tail(15).plot.barh()

plt.title("Variables más importantes")

plt.show()