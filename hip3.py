# Existem padrões sazonais nos pedidos de diferentes especialidades ao longo do ano.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Carregar os dados
df = pd.read_csv('orders.csv')

# Remover quaisquer linhas nulas
df = df.dropna()

# Convertendo a coluna de data para tipo datetime
df['date'] = pd.to_datetime(df['date'])

# Selecionar dados relevantes para análise de padrões sazonais
X = df[['date', 'spec', 'successful_orders', 'fail_orders']].copy()  # Cria uma cópia para evitar o SettingWithCopyWarning

# Extrair mês e ano como features adicionais usando .loc
X.loc[:, 'month'] = X['date'].dt.month
X.loc[:, 'year'] = X['date'].dt.year

# Agrupamento por especialidade e mês
grouped_data = X.groupby(['spec', 'month']).agg({
    'successful_orders': 'sum',
    'fail_orders': 'sum'
}).reset_index()

# Normalizar os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(grouped_data[['successful_orders', 'fail_orders']])

# Aplicar KMeans para identificar clusters
kmeans = KMeans(n_clusters=3, random_state=42)
grouped_data['cluster'] = kmeans.fit_predict(scaled_data)

# Plotar resultados do agrupamento
plt.figure(figsize=(12, 8))
sns.scatterplot(x='successful_orders', y='fail_orders', hue='cluster', data=grouped_data, palette='Set1', s=100)
plt.title('Agrupamento de Especialidades por Padrões de Pedidos')
plt.xlabel('Pedidos Bem-sucedidos')
plt.ylabel('Pedidos com Falha')
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
