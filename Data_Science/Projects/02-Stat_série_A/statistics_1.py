import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt
from collections import Counter


df = pd.read_csv('./assets/brasileirao_serie_a.csv.gz')
df = df.dropna()
df = df.drop(df[df['faltas_mandante'] > 50].index)


def media(data):
    try:
        return data.sum()/len(data) - 1
    except:
        return sum(data)/len(data) - 1


def desvio(val, media_):
    return val - media_


def variancia(desvio_):
    return desvio_ ** 2


def desvio_padrao(variancia_):
    if 0 > variancia_:
        variancia_ *= -1
        variancia_ = -sqrt(variancia_)
    else:
        return sqrt(variancia_)
    return variancia_


def padrao(desvio, desvio_padrao_):
    return desvio / desvio_padrao_


gols_mandante = []
var = sum([variancia(desvio(x, media(df['gols_mandante'])))
           for x in df['gols_mandante']]) / len(df['gols_mandante']) - 1
for val in df['gols_mandante']:
    gols_mandante.append(padrao(
        desvio(val, media(df['gols_mandante'])),
        desvio_padrao(var)
    ))

faltas_mandante = []
var = sum([variancia(desvio(x, media(df['faltas_mandante'])))
           for x in df['faltas_mandante']]) / len(df['faltas_mandante']) - 1
for val in df['faltas_mandante']:
    faltas_mandante.append(padrao(
        desvio(val, media(df['faltas_mandante'])),
        desvio_padrao(var)
    ))

corr = []
for c in range(0, len(faltas_mandante)):
    corr.append(
        gols_mandante[c] * faltas_mandante[c]
    )
# print(media(corr))


plt.title('Número de gols')
plt.bar('Time mandante', df['gols_mandante'].sum())
plt.bar('Time visitante', df['gols_visitante'].sum())
plt.show()


fig, axs = plt.subplots(2)
fig.suptitle('Histograma n de gols')


for k, v in Counter(df['gols_visitante']).items():
    axs[0].bar(k, v, color='orange')
for k, v in Counter(df['gols_mandante']).items():
    axs[1].bar(k, v, color='blue')


plt.title('Faltas e Gols do time mandante')
plt.scatter([x for x in range(len(df['faltas_mandante']))],
            df['faltas_mandante'], color='red')
plt.scatter([x for x in range(len(df['faltas_mandante']))],
            df['gols_mandante'], color='green')
