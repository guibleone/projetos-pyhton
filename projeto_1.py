# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# %matplotlib inline


#Geração do modelo
X = np.array([
#  Farinha Açúcar Óleo Sal Mateiga Chocolate#     
[100	,	10	,	0	,	0	,	5	,	5	],
[100	,	10	,	0	,	0	,	6	,	6	],
[100	,	10	,	0	,	0	,	7	,	7	],
[100	,	15	,	0	,	0	,	8	,	8	],
[100	,	15	,	2	,	1	,	9	,	9	],
[100	,	15	,	2	,	1	,	10	,	10	],
[100	,	20	,	2	,	1	,	5	,	12	],
[150	,	20	,	2	,	1	,	6	,	10	],
[150	,	20	,	4	,	1	,	7	,	8	],
[150	,	22	,	4	,	2	,	8	,	5	],
[150	,	22	,	4	,	2	,	9	,	6	],
[150	,	24	,	4	,	2	,	5	,	8	],
[150	,	24	,	4	,	2	,	8	,	10	]	])

Y = np.array([
[	4.1	]	,
[	4.9	]	,
[	5.4	]	,
[	6.4	]	,
[	6.9	]	,
[	7.4	]	,
[	9.0	]	,
[	8.0	]	,
[	7.0	]	,
[	5.7	]	,
[	5.9	]	,
[	7.1	]	,
[	7.8	]	])

#### Criar Data Frame
df = pd.DataFrame(np.append(X,Y,axis = 1))
df

###### Indicar principais descritores estatístico das séries
#df.describe()

###### descobrir se há correlação entre as variáveis e a resposta
#np.round(df.corr(),2)
#sns.heatmap(df.corr(),cmap='coolwarm')
#plt.show()


#### Visualize Chocolate vs Nota da bolacha, observar a correlação
#plt.scatter(X[:,5],Y[:,0],c = 'g',s=15,alpha=0.5)
#plt.xlabel('Chocolate')
#plt.ylabel('Sabor')
#plt.show()

# Visualize Manteiga vs Nota da bolacha, observar a correlação
#plt.scatter(X[:,4],Y[:,0],c = 'g',s=15,alpha=0.5)
#plt.xlabel('Manteiga')
#plt.ylabel('Sabor')
#plt.show()

##### ajustando para regressão linear para ter constante (intercepto) b
lm = LinearRegression(fit_intercept=True) # y = ax + b
lm.fit(X,Y)

##### constante ou intercept.
#lm.intercept_

### coeficientes dos parametros
#lm.coef_

#  Farinha Açúcar Óleo Sal Mateiga Chocolate# 
# Sabor = k + aF + bA + cO + dS + eM + fC

### calcular as respostas previstas
predY = lm.predict(X)
predY

# Gráfico  real Y x  Y previsto.
#plt.scatter(Y,predY,c = 'blue', s=15, alpha=0.5)
#plt.xlabel('Sabor real')
#plt.ylabel('Sabor previsto')
#plt.show()


# Coeficiente de determinação (R^2):
#lm.score(X,Y)

# Calculate residual.
residuo = Y - predY
residuo


# Observar se o resíduo tem distribuição aleatória com média zero
#plt.scatter(Y,residuo,c = 'red', s=15, alpha=0.5)
#plt.xlabel('Y')
#plt.ylabel('Resíduo')
#plt.title('Resíduo')
#plt.show()

# Observar se os resíduos tem distribuição normal e média zero
#sns.distplot(residuo, bins=50, color='green').set_title("Residual Histogram")
#plt.show()


#Criar tabela de análise estatística
import statsmodels.api as sm
results = sm.OLS(Y, X).fit()
A = np.identity(len(results.params))
A = A[1:,:]
#print(results.f_test(A))
print(results.summary())

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
# %matplotlib inline

#pulos
X  = np.array([
[0],
[20],
[40],
[60],
[80],
[100],
[120],
[140] ])

#batimento cardiaco
Y = np.array([
[86],
[115],
[133],
[130],
[125],
[135],
[143],
[138] ])

# criar data frame

df = pd.DataFrame(np.append(X,Y, axis=1))
#df

#df.describe()

#plt.scatter(X[:,0],Y[:,0],c = 'g',s=15,alpha=0.5)
#plt.xlabel('Pulo')
#plt.ylabel('Batimento')
#plt.show()

lm = LinearRegression(fit_intercept=True) # y = ax + b
lm.fit(X,Y)

#lm.intercept_

#lm.coef_

predY = lm.predict(X)
predY

# Gráfico  real Y x  Y previsto.
#plt.scatter(Y,predY,c = 'blue', s=15, alpha=0.5)
#plt.xlabel('Batimento real')
#plt.ylabel('Batimento previsto')
#plt.show()

# Coeficiente de determinação (R^2):
#lm.score(X,Y)

# Calculate residual.
#residuo = Y - predY
#residuo

# Observar se o resíduo tem distribuição aleatória com média zero
#plt.scatter(Y,residuo,c = 'red', s=15, alpha=0.5)
#plt.xlabel('Y')
#plt.ylabel('Resíduo')
#plt.title('Resíduo')
#plt.show()

# Observar se os resíduos tem distribuição normal e média zero
#sns.distplot(residuo, bins=50, color='green').set_title("Residual Histogram")
#plt.show()

import statsmodels.api as sm

X = sm.add_constant(X)

results = sm.OLS(Y, X).fit()
A = np.identity(len(results.params))
A = A[1:,:]
print(results.f_test(A))
print(results.summary())