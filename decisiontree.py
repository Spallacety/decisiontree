import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

'''
primeiro parametro de x (tempo):
0 = sol
1 = nublado
2 = chuva

segundo parametro de x (temperatura):
0 = baixa
1 = media
2 = alta

'Type', 'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash  ', 'Magnesium', 'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins', 'Color_intensity', 'Hue', 'Diluted_wines', 'Proline'


terceiro parametro de x (umidade):
0 = baixa
1 = media
2 = alta

quarto parametro de x (vento):
0 = nao
1 = sim

y (joga):
0 = nao
1 = sim
'''

x = [
[0, 2, 1, 0],
[0, 2, 2, 1],
[1, 2, 2, 0],
[2, 0, 2, 0],
[2, 0, 1, 0],
[2, 0, 0, 1],
[1, 0, 0, 1],
[0, 1, 2, 0],
[0, 0, 0, 0],
[2, 1, 1, 0],
[0, 1, 0, 1],
[1, 1, 2, 1],
[1, 2, 0, 0],
[2, 0, 2, 1]
]

y = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0]

pred_nublado = [[1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 2, 0], [1, 0, 2, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 2, 0, 0], [1, 2, 0, 1], [1, 2, 1, 1], [1, 2, 2, 1]]
pred_chuva_vento_nao = [[2, 0, 0, 0], [2, 0, 1, 0], [2, 1, 0, 0], [2, 1, 1, 0]]
pred_chuva_vento_sim = [[2, 0, 0, 1], [2, 0, 1, 1], [2, 1, 0, 1], [2, 1, 1, 1]]
pred_sol_umidade_baixa = [[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 2, 0, 1]]
pred_sol_umidade_media = [[0, 0, 1, 0], [0, 1, 1, 0], [0, 2, 1, 0], [0, 0, 1, 1], [0, 1, 1, 1], [0, 2, 1, 1]]
pred_sol_umidade_alta = [[0, 0, 2, 0], [0, 1, 2, 0], [0, 2, 2, 0], [0, 0, 2, 1], [0, 1, 2, 1], [0, 2, 2, 1]]

dt = DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(x, y)

print(dt.predict(pred_nublado))
print(dt.predict(pred_chuva_vento_nao))
print(dt.predict(pred_chuva_vento_sim))
print(dt.predict(pred_sol_umidade_baixa))
print(dt.predict(pred_sol_umidade_media))
print(dt.predict(pred_sol_umidade_alta))

train_df = pd.read_csv('random.csv')
train_df = train_df.drop(['country_off_res', 'age_desc', 'used_app_before', 'relation'], 1)

train_df.loc[train_df['ethnicity'].isnull(), 'ethnicity'] = 'Unknown'
train_df['gender'] = train_df['gender'].map({'m' : 0, 'f' : 1}).astype(int)
train_df['jaundice'] = train_df['jaundice'].map({'no' : 0, 'yes' : 1}).astype(int)
train_df['pdd'] = train_df['pdd'].map({'no' : 0, 'yes' : 1}).astype(int)
train_df['ethnicity'] = train_df['ethnicity'].map({'White-European' : 0 ,'Middle-Eastern' : 1 ,'Hispanic' : 2 ,'Asian' : 3 ,'Black' : 4 ,'South-Asian' : 5 ,'Latino' : 6 ,'Pasifika' : 7 ,'Turkish' : 8, 'Others' : 9, 'Unknown': 10}).astype(int)
train_df['classification'] = train_df['classification'].map({'NO' : 0, 'YES' : 1}).astype(int)

train_df = train_df.dropna(how='any',axis=0)

X_df = train_df [['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'age', 'ethnicity', 'gender', 'jaundice', 'pdd', 'result']]
Y_df = train_df ['classification']

Xdummies_df = pd.get_dummies(X_df)

X = Xdummies_df.values
Y = Y_df.values

training_percentage = 0.7
test_percentage = 0.3

training_size = int(training_percentage * len(Y))
test_size = int(test_percentage * len(Y))

training_data = X[:training_size]
training_markers = Y[:training_size]

training_end = training_size + test_size

test_data = X[training_size:training_end]
test_markers = Y[training_size:training_end]

dt = DecisionTreeClassifier(criterion='entropy')
dt = dt.fit(training_data, training_markers)

print(len(test_markers))

correct = 0
size = len(test_markers)
for i in range(size):
  if test_markers[i] == dt.predict(test_data)[i]:
    correct += 1

print(correct)