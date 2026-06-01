import pickle
import pandas as pd
import numpy as np


# 1. CARREGAR DADOS

modelo = pickle.load(open('bank_rf.pkl', 'rb'))
le_classe = pickle.load(open('le_classe.pkl', 'rb'))
colunas_treino = pickle.load(open('colunas_treino.pkl', 'rb'))


# 2. NOVO DADO

novo_dado = [25, "services", "single", "basic.6y", "unknown", "yes", "no", 
             "cellular", "jul", "thu", 301, 1, 999, 0, "nonexistent", 
             1.4, 93.918, -42.7, 4.958, 5228.1]

# Nomes das colunas
colunas_originais = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
                     'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
                     'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx', 
                     'cons.conf.idx', 'euribor3m', 'nr.employed']


# 3. CRIAR DATAFRAME

novo_cliente = pd.DataFrame([novo_dado], columns=colunas_originais)

# 4. APLICAR ONE-HOT ENCODING

novo_cliente = pd.get_dummies(novo_cliente, drop_first=True)

# 5. ALINHAR COM AS COLUNAS DO TREINO

# Adicionar colunas que faltam (preencher com 0)
for col in colunas_treino:
    if col not in novo_cliente.columns:
        novo_cliente[col] = 0

# Remover colunas que existem no novo dado mas não no treino
for col in novo_cliente.columns:
    if col not in colunas_treino:
        novo_cliente = novo_cliente.drop(columns=[col])

# Reordenar colunas na MESMA ordem do treino
novo_cliente = novo_cliente[colunas_treino]

# classificar
resultado_numero = modelo.predict(novo_cliente)
resultado_texto = le_classe.inverse_transform(resultado_numero)

print(f" RESULTADO: {resultado_texto[0]}")