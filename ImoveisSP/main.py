import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.preprocessing import OneHotEncoder

px.set_mapbox_access_token(open("mapbox_token").read())

df = pd.read_csv("sao-paulo-properties-april-2019.csv")

df_rent = df[df["Negotiation Type"]=="rent"]
df_sale = [df["Negotiation Type"]=="sale"]


df_cleaned = df_rent.drop(["New", "Property Type", "Negotiation Type"], axis=1)

cat_encoder = OneHotEncoder()

housing_cat_1hot = cat_encoder.fit_transform(df_cleaned[["District"]])


one_hot = pd.get_dummies(df_cleaned["District"])

df = df_cleaned.drop('District',axis = 1)
df = df.join(one_hot)

#treino de mutiplos modelos
from sklearn.model_selection import train_test_split

Y = df["Price"]
X = df.loc[:, df.columns != "Price"]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)


from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)

some_data = X.iloc[:5]
some_labels = Y.iloc[:5]

print("Predictions:", lin_reg.predict(some_data))
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error

preds = lin_reg.predict(x_train)
lin_mse = mean_squared_error(y_train, preds)

lin_rmse = np.sqrt(lin_mse)


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train, y_train)

from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5, 
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(x_train, y_train)

final_model = grid_search.best_estimator_
final_predictions = final_model.predict(x_test)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print(final_rmse)
#erro de preços de 1599.00