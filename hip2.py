# Fornecedores com mais pedidos bem-sucedidos têm menos pedidos com falha.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Carregar os dados
df = pd.read_csv('orders.csv')

# Remover quaisquer linhas nulas
df = df.dropna()

# Visualizar as primeiras linhas do dataframe para entender a estrutura dos dados
print(df.head())

# Análise Exploratória
# Plotando relação entre pedidos bem-sucedidos e pedidos com falha
plt.figure(figsize=(10, 6))
sns.scatterplot(x='successful_orders', y='fail_orders', data=df)
plt.title('Relação entre Pedidos Bem-sucedidos e Pedidos com Falha')
plt.xlabel('Pedidos Bem-sucedidos')
plt.ylabel('Pedidos com Falha')
plt.grid(True)
plt.show()

# Modelagem Preditiva
# Selecionar features e target
X = df[['successful_orders']]
y = df['fail_orders']

# Dividir dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo de regressão linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar desempenho do modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Erro quadrático médio (MSE): {mse}')

# Plotar resultados da regressão
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regressão Linear')
plt.title('Regressão Linear: Pedidos Bem-sucedidos vs Pedidos com Falha')
plt.xlabel('Pedidos Bem-sucedidos')
plt.ylabel('Pedidos com Falha')
plt.legend()
plt.grid(True)
plt.show()
