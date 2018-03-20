import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

'''
primeiro parametro de x (tempo):
0 = sol
1 = nublado
2 = chuva

segundo parametro de x (temperatura):
0 = baixa
1 = media
2 = alta

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

#print(dt.predict(pred_nublado))
#print(dt.predict(pred_chuva_vento_nao))
#print(dt.predict(pred_chuva_vento_sim))
#print(dt.predict(pred_sol_umidade_baixa))
#print(dt.predict(pred_sol_umidade_media))
#print(dt.predict(pred_sol_umidade_alta))

# new database

def fit_and_predict(name, model, training_data, training_markers, test_data, test_markers):
  model.fit(training_data, training_markers)
  result = model.predict(test_data)

  correct = 0
  size = len(test_markers)
  for i in range(size):
    if test_markers[i] == result[i]:
      correct += 1

  print('%s: %.2f%% correctly predict' %(name, (correct*100/size)))

train_df = pd.read_csv('wine4.csv')

train_df['fixed_acidity'] = train_df['fixed_acidity'].astype(float)
train_df['volatile_acidity'] = train_df['volatile_acidity'].astype(float)
train_df['citric_acid'] = train_df['citric_acid'].astype(float)
train_df['residual_sugar'] = train_df['residual_sugar'].astype(float)
train_df['chlorides'] = train_df['chlorides'].astype(float)
train_df['free_sulfur_dioxide'] = train_df['free_sulfur_dioxide'].astype(int)
train_df['total_sulfur_dioxide'] = train_df['total_sulfur_dioxide'].astype(int)
train_df['density'] = train_df['density'].astype(float)
train_df['pH'] = train_df['pH'].astype(float)
train_df['sulphates'] = train_df['sulphates'].astype(float)
train_df['alcohol'] = train_df['alcohol'].astype(float)
train_df['quality'] = train_df['quality'].astype(int)

X_df = train_df [['fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides', 'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol']]
Y_df = train_df ['quality']

Xdummies_df = pd.get_dummies(X_df)

X = Xdummies_df.values
Y = Y_df.values

training_percentage = 0.85
test_percentage = 0.15

training_size = int(training_percentage * len(Y))
test_size = int(test_percentage * len(Y))

training_data = X[:training_size]
training_markers = Y[:training_size]

training_end = training_size + test_size

test_data = X[training_size:training_end]
test_markers = Y[training_size:training_end]

modelDecisionTree = DecisionTreeClassifier(criterion='entropy')

fit_and_predict("DecisionTree", modelDecisionTree, training_data, training_markers, test_data, test_markers)

#os resultados são diferentes porque a cada vez ele monta uma arvore diferente