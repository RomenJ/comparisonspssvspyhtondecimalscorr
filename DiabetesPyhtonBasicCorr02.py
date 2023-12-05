import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.impute import SimpleImputer
import numpy as np
from scipy import stats

# origin_dataset: https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset
datos = pd.read_csv('diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
print('Nº de casos:', len(datos))

X = datos[['HighChol','BMI', 'Fruits','Age','Income','HighBP', 'GenHlth','Sex','Education']]
Y = datos['Diabetes_binary']

# Dividir el conjunto de datos en entrenamiento y prueba (70% entrenamiento, 30% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Imputa los valores faltantes en Y_train
imputer = SimpleImputer(strategy='most_frequent')
Y_train = imputer.fit_transform(Y_train.values.reshape(-1, 1))
Y_train = Y_train.flatten()

# Imputa los valores faltantes en Y_test
Y_test = imputer.transform(Y_test.values.reshape(-1, 1))
Y_test = Y_test.flatten()

# Imputa los valores faltantes en X_train
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# Imputa los valores faltantes en X_test
X_test = imputer.transform(X_test)

correlation_matrix_with_target = datos.corr()

# Calcular la significancia estadística
p_values = datos.corr(method=lambda x, y: stats.pearsonr(x, y)[1])  # Método de correlación de Pearson

# Crear un DataFrame para mostrar la matriz de correlaciones con los niveles de significación
correlation_df = pd.DataFrame(correlation_matrix_with_target, columns=correlation_matrix_with_target.columns, index=correlation_matrix_with_target.columns)

# Agregar los asteriscos según la significancia estadística
alpha_005 = 0.05
alpha_001 = 0.01
significance_markers_005 = (p_values < alpha_005).astype(str).replace({'True': '*'})
significance_markers_001 = (p_values < alpha_001).astype(str).replace({'True': '%'})
correlation_df = correlation_df.astype(str) + significance_markers_005 + significance_markers_001

# Mostrar la matriz de correlaciones con los niveles de significación
print("Matriz de Correlaciones con la variable objetivo :")
print(correlation_df)

correlation_matrix_with_target = datos.corr()

# Graficar la matriz de correlaciones con la variable objetivo como un mapa de calor
plt.figure(figsize=(12, 10))
heatmap = plt.imshow(correlation_matrix_with_target, cmap='Blues', interpolation='none')
plt.colorbar(heatmap)
plt.xticks(range(len(correlation_matrix_with_target)), correlation_matrix_with_target.columns, rotation=90)
plt.yticks(range(len(correlation_matrix_with_target)), correlation_matrix_with_target.columns)

# Agregar etiquetas con los valores de correlación en cada celda del mapa de calor
for i in range(len(correlation_matrix_with_target)):
    for j in range(len(correlation_matrix_with_target)):
        plt.text(j, i, "{:.1f}".format(correlation_matrix_with_target.iloc[i, j]),
                 ha='center', va='center', color='black', fontsize=8)

plt.title('Matriz de Correlaciones con la variable objetivo')
plt.show()