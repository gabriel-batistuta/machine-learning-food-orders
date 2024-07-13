# A especialidade mais popular varia significativamente entre as cidades?

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar os dados
df = pd.read_csv('orders.csv')

# Remover quaisquer linhas nulas
df = df.dropna()

# Visualizar as primeiras linhas do dataframe para entender a estrutura dos dados
print(df.head())

# Estatísticas descritivas
print(df.describe())

# Selecionar apenas as colunas numéricas para calcular a matriz de correlação
numeric_columns = df.select_dtypes(include=np.number).columns
correlation_matrix = df[numeric_columns].corr()
print(correlation_matrix)

# Selecionar features e target
X = df[['vendor_id', 'chain_id', 'city_id', 'successful_orders']]
y = df['fail_orders']

# Pré-processamento de variáveis
numeric_features = ['vendor_id', 'chain_id', 'city_id', 'successful_orders']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
    ])

# Definir pipeline completo com modelo
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', RandomForestClassifier(random_state=42))])

# Dividir dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo
pipeline.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliar desempenho do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {accuracy}')
