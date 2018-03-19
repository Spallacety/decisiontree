import numpy as np
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