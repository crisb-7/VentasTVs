# %% [markdown]
# # Series de tiempo no estacionarias - Tendencia {-}
# ## Módulo 5 - TC3007C {-}
# ### Cristofer Becerra Sánchez - A01638659 {-}

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
ventas = np.array([4.8, 4.1, 6.0, 6.5, 5.8, 5.2, 6.8, 7.4, 6.0, 5.6, 7.5, 7.8, 6.3, 5.9, 8.0, 8.4])
t = np.arange(1, len(ventas)+1)
t_trim = np.array([y for x in range(4) for y in range(1,5)])

# %% [markdown]
# # Visualización de ventas

# %%
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.titlesize"] = 16

fig,axes = plt.subplots(1,1, figsize=(10,5))
title, xlabel, ylabel = "Ventas de televisores", "Trimestre [n]", "Ventas [miles]"
axes.plot(t, ventas, marker="o", color = "#0072B2", linewidth=2, label="Ventas")
axes.set_title(title)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
axes.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Análisis de tendencia y estacionalidad
# 
# Descomposición de la serie en sus 3 componentes

# %% [markdown]
# ## Media móvil y media móvil centrada

# %%
def media_movil(x, w):
    # x --> arreglo
    # w --> ventana
    rolling = np.zeros(len(x)-w+1)
    for i in range(len(rolling)):   
        rolling[i] = np.mean(x[i:i+w])
    return rolling

# %%
# Obtener media movil
mm = media_movil(ventas, 4)
print(mm)
print(len(mm))

# %%
# Obtener media movil centrada
mmC = media_movil(mm, 2)
print(mmC)
print(len(mmC))

# %%
def centrar_array(main, sub):
    n1 = round((len(main) - len(sub)))
    n2 = len(main)
    centered = [None for x in main]
    centered[n1:n2] = sub
    return np.array(centered)

# %%
mediaMovil = centrar_array(ventas, mm)
mediaMovilCentrada = centrar_array(ventas, mmC)
print(mediaMovil)
print(len(mediaMovil))
print(mediaMovilCentrada)
print(len(mediaMovilCentrada))

# %%


# %% [markdown]
# ### Visualización

# %%
title, xlabel, ylabel = "Ventas de televisores", "Trimestre [n]", "Ventas [miles]"

fig,axes = plt.subplots(1,1, figsize=(10,5))
axes.plot(t, ventas, marker="o", color = "#0072B2", linewidth=2, label="Ventas")
axes.plot(t[t>3], mm, marker="v", color="#E69F00", linewidth=2, label="Media Móvil")
axes.plot(t[t>4], mmC, marker="p", linestyle="--", color="#009E73", linewidth=1.5, label="Media Móvil Centrada")
axes.set_title(title)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# ## Valores irregulares y desestacionalizados

# %% [markdown]
# Valores irregulares

# %%
n1 = round((len(ventas) - len(mmC))/2)
n2 = round((len(ventas) + len(mmC))/2)
irreg = np.divide(ventas[n1:n2], mmC)
irregC = centrar_array(ventas, irreg)
print(irreg)

# n = 4
# print(irreg[np.where(t_trim[n1:n2] == n)[0]])
# np.unique(t_trim[n1-1:n2+1], return_counts=True)
# len(t_trim[n1-1:n2+1])
# len(irreg)

# %% [markdown]
# Índice estacional

# %%
ind_estac = []
for i in range(1, 5):
    ind_estac.append(irreg[np.where(t_trim[n1:n2] == i)[0]].mean())

print(ind_estac)

# %% [markdown]
# Vector de longitud de las ventas con el índice estacional en su respectiva estación

# %%
ind_vec = np.array([ind for x in range(4) for ind in ind_estac])
print(ind_vec)

# %% [markdown]
# Ventas Tendencias

# %%
desest = np.divide(ventas, ind_vec)
print(desest)

# %%
title, xlabel, ylabel = "Ventas de televisores", "Trimestre [n]", "Ventas [miles]"

fig,axes = plt.subplots(1,1, figsize=(10,5))
axes.plot(t, ventas, marker="o", color = "#0072B2", linewidth=2, label="Ventas")
axes.plot(t, desest, marker="^", color="#D55E00", linewidth=2, label="Tendencia")
axes.set_title(title)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
plt.legend()
plt.grid()
plt.show()


# %% [markdown]
# # Regresión lineal de la serie Tendencia

# %%
from statsmodels.formula.api import ols
from pandas import DataFrame

# %%
df = DataFrame({"Trimestre":t, "Ventas":desest})

# %%
model = ols("Ventas ~ Trimestre", data=df)
linear_reg = model.fit()
linear_reg.summary()


# %%
b0, b1 = linear_reg.params
print(b0, b1)

# %%
y = linear_reg.predict()
e = linear_reg.resid

# %%
fig,axes = plt.subplots(1,1, figsize=(10,5))
axes.plot(t, desest, marker="^", color = "#D55E00", linewidth=2, label="Tendencia")
axes.plot(t, y, linestyle="--", label=r"Regresión [$\hat{Y}_i$]", color="#56B4E9")
axes.set_title("Venta de televisores - Regresión de la tendencia")
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
axes.set_ylim([3.8, 8.5])
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Verificación del modelo de regresión

# %% [markdown]
# ## Coeficiente de determinación $R^2$

# %% [markdown]
# Se comienza la verificación del modelo revisando el coeficiente de determinación. Al imprimirlo,

# %%
print("R^2 - ", linear_reg.rsquared)
print("R^2 Ajustado - ", linear_reg.rsquared_adj)

# %% [markdown]
# parece ser que éste es elevado y por lo tanto es probable que, en efecto, el modelo (tendencia) sea significativo.

# %% [markdown]
# ## Significancia de $\beta_1$

# %% [markdown]
# Se prosigue con la verificación de la significancia del coeficiente que otorga la tendencia lineal,

# %%
import scipy.stats as stats

# %%
n = len(df)                             # Tamaño de muestra
k = 1                                   # Numero de variables independientes del modelo

sb1 = e.std()/(np.sqrt(n-2)*t.std())    # Estimador de Beta 1
ts = (b1)/(sb1)                         # Estadístico de prueba, t

p_value = 2*(1 - stats.t.cdf(x = abs(ts), df = n-k-1))

print("t de pureba: " + str(ts))
print("p-value: " + str(p_value))


# %% [markdown]
# y, tras calcular el estimador del coeficiente y su respectivo estadístico de prueba $t^*$, se observa que tiene el mismo valor que el arrojado por la verificación del modelo de Statsmodels, y el p-value asociado es cercano a cero. Por lo tanto, es posible descartar la hipótesis nula de que el coeficiente $\beta_1 = 0$.

# %% [markdown]
# ## Análisis de residuos

# %% [markdown]
# ### Normalidad

# %% [markdown]
# Se realiza una prueba de normalidad de Shapiro-Wilk sobre los residuos

# %%
s, p = stats.shapiro(e)
print("Estadístico W:", s)
print("p-value", p)


# %% [markdown]
# Se obtiene un valor bastante mayor a un $\alpha = 0.05$, por lo cual no se rechaza la hipótesis nula de que los residuos siguen una distribución normal. Se visualiza lo anterior con un QQ-plot:

# %%
import seaborn as sns

# %%
fig, axes = plt.subplots(1,2, figsize=(10,5))

stats.probplot(e, plot = axes[0])
axes[0].get_lines()[0].set_marker('o')
axes[0].get_lines()[0].set_color('midnightblue')
axes[0].get_lines()[0].set_markeredgecolor('k')
axes[0].get_lines()[1].set_color('darkorange')
axes[0].get_lines()[1].set_linestyle('--')
axes[0].set_title("Gráfico Q-Q: residuos")
axes[0].set_xlabel("Cuantiles teóricos")
axes[0].set_ylabel("Valores observados")
axes[0].grid()

sns.histplot(x = e, kde = True, stat = "density", color = "#009E73", ax = axes[1])
axes[1].set_title("Distribución de residuos")
axes[1].set_xlabel("Residuo")
axes[1].set_ylabel("Frecuencia")

plt.tight_layout()
plt.grid()
plt.show()

# %% [markdown]
# ### Homocedasticidad

# %% [markdown]
# Se grafica la varianza de los residuos para ver si siguen alguna tendencia importante

# %%
fig,axes = plt.subplots(1,1, figsize=(9,4))
sns.regplot(x = y, y = e, color = "midnightblue", ax = axes)
axes.set_title("Homocedasticidad de los residuos")
axes.set_xlabel("Predicción de venta [miles]")
axes.set_ylabel(r"Residuo $e_n$")
plt.grid()
plt.show()

# %% [markdown]
# A pesar de que hay pocos datos, parece ser que la regresión de Seaborn indica una nula tendencia, no obstante, debe realizarse una prueba de hipótesis de independencia; es decir, que la media de los residuos no sea diferente de cero:

# %%
e_tstat, e_ind_pvalue = stats.ttest_1samp(e, popmean=0)
print("Estadístico de prueba t:", e_tstat)
print("Media de los residuos", e.mean())
print("p-value:", e_ind_pvalue)

# %% [markdown]
# ya que se obtuvo un p-value muy cercano a 1, no se puede rechazar la hipótesis nula que establece que la media de los residuos es igual a cero; por lo tanto, se comprueba la condición de homocedasticidad.

# %% [markdown]
# # Errores en la predicción de la serie de tiempo

# %%
# MSE - Mean Squared Error
# MAPE - Mean absolute percentage error
predicciones = np.multiply(y, ind_vec)

MSE = ((ventas - predicciones)**2).sum()/n
MAPE = np.abs(np.divide(ventas - predicciones, ventas)).sum()/n

print("Cuadrado medio del error:", round(MSE, 4))
print("Promedio de los errores porcentuales:",round(MAPE, 4))

# %%
title, xlabel, ylabel = "Ventas de televisores", "Trimestre [n]", "Ventas [miles]"

fig,axes = plt.subplots(1,1, figsize=(10,5))
axes.plot(t, ventas, marker="o", color = "#0072B2", linewidth=2, label="Ventas")
axes.plot(t, predicciones, marker="^", color="k", linewidth=2, label="Predicción")
axes.set_title(title)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
plt.legend()
plt.grid()
plt.show()

# %% [markdown]
# # Pronóstico del año siguiente

# %%
t_pred = np.array([17,18,19,20])

tendencia = b0 + b1*t_pred
pred = np.multiply(tendencia, np.array(ind_estac))

print("Predicciones Año 5:", pred.round(4))

# %%
t_pred_viz = [x for x in range(16, 21)]
pred_viz = [ventas[-1]]
pred_viz.extend([x for x in pred])
print(t_pred_viz)
print(pred_viz)

# %%
title, xlabel, ylabel = "Ventas de televisores", "Trimestre [n]", "Ventas [miles]"

fig,axes = plt.subplots(1,1, figsize=(10,5))

axes.plot(t, ventas, marker="o", color = "#0072B2", linewidth=2, label="Ventas")
axes.plot(t, desest, marker="^", color="#D55E00", linewidth=1.5, label="Tendencia")
axes.plot(t_pred_viz, pred_viz, marker="D", color="#5988A2", linestyle="--", linewidth=2, label="Pronóstico Año 5")
axes.set_title(title)
axes.set_xlabel(xlabel)
axes.set_ylabel(ylabel)
plt.legend()
plt.grid()
plt.show()


