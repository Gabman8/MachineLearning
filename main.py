from cmath import sqrt
from xml.parsers.expat import model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# The file is semicolon-delimited; default comma parsing causes field-count errors.
airbnb_data = pd.read_csv("./airbnb-listings-extract.csv", sep=";", decimal='.')
columns = [
    # tamaño del inmueble
    'Accommodates',
    'Bathrooms',
    'Bedrooms',
    'Beds',
    'Square Feet',
    # tipo de propiedad
    'Property Type',
    'Room Type',
    'Bed Type',
    # ubicación
    'Latitude',
    'Longitude',
    'Neighbourhood Cleansed',
    # reputación
    'Review Scores Rating',
    'Review Scores Accuracy',
    'Review Scores Cleanliness',
    'Review Scores Checkin',
    'Review Scores Communication',
    'Review Scores Location',
    'Review Scores Value',

    # popularidad
    'Number of Reviews',
    'Reviews per Month',

    # disponibilidad
    'Availability 30',
    'Availability 60',
    'Availability 90',
    'Availability 365',

    # host
    'Host Listings Count',
    'Host Total Listings Count',
    'Host Response Rate',
    'Host Acceptance Rate'
]

X = airbnb_data[columns]
y = airbnb_data['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model.fit(X_train, y_train)
pred = model.predict(X_test)