from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Cargar el conjunto de datos
boston = load_boston()
X, y = boston.data, boston.target

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Guardar el modelo entrenado
joblib.dump(model, 'linear_regression_model.pkl')
