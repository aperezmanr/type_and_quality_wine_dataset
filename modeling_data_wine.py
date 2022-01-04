#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  27 18:41:47 2021

@author: Alejandro Pérez Manrique
"""

# Importamos librerías que emplearemos a lo largo de todo
# el proyecto. Este chunck será modificado 
# tantas veces como veamos que necesitamos añadir una librería
# para el tratamiento de los datos.

# Carga de las librerías necesarias (se irán incorporando a medida que
# las necesitemos)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.tree import export_text
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import resample
# Empleamos estilo gráfico de ggplot (R)
style.use('ggplot') or plt.style.use('ggplot')
# Suprimimos los avisos de aviso para evitar ensuciar nuestra visualización
warnings.filterwarnings('ignore')

df_wine = pd.read_csv('data/df_wine_cleaned.csv')

# Transformamos de nuevo la variable target en 0, 1, 2, 3
df_wine.loc[df_wine.rating == 'D', 'quality'] = 0
df_wine.loc[df_wine.rating == 'C', 'quality'] = 1
df_wine.loc[df_wine.rating == 'B', 'quality'] = 2
df_wine.loc[df_wine.rating == 'A', 'quality'] = 3

# Seleccionamos nuestra variable objetivo
target = df_wine['Type'] 

# Eliminamos las variables que no nos sirven
df_wine = df_wine.drop(['Type', 'rating'], axis=1)

# Creación de nuestras muestras de entrenamiento y test
X_train, X_test, y_train, y_test = train_test_split(
    df_wine, target, test_size=0.30, random_state=42, stratify=target)

# Comprobación del tamaño de nuestra muestra de entrenamiento
print(y_train)

# Nos aseguramos de que los valores numéricos de target sean numéricos
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Deberemos importar la herramienta StandardScaler
standard = StandardScaler()
X_train = standard.fit_transform(X_train)
X_test = standard.transform(X_test)

# Necesitaremos importar DecisionTreeClassifier, f1_score,
# recall_score, make_scorer, metrics y precision_score para
# trabajar con el código que viene a continuación. Además,
# crearemos una función que nos ayude a ejecutar el árbol de
# decisiones las veces que deseemos sin tener que copiar código.
def arbol_decisiones(max_depths, X_train, y_train, X_test, y_test):
    """" Función que ejecuta un árbol de decisiones.
    
    La función ejecutará un árbol de decisiones en función de unos
    parámetros introducidos. Estos parámetros deberán entenderse que
    son los de entrenamiento y los de testeo, es por ello que se
    recomienda primero tratar con el set de datos para conocer que
    se introduce a la función.
    
      Argumentos empleados (por orden):
    max_depths -- Profundidad deseada del árbol de decisiones
    X_train    -- Conjunto de parámetros de entrenamiento
    y_train    -- Conjunto de valores objetivos de entrenamiento
    X_test     -- Conjunto de parámetros de testeo
    y_train    -- Conjunto de valores objetivo de testeo

    Retornará:
    train_errors  -- Lista con los errores cuadráticos medios de entrenamiento
    test_errors   -- Lista con los errores cuadráticos medios de testeo
    
    """
    # Guardaremos los errores cuadráticos medios
    train_errors = []
    test_errors = []

    for max_depth in max_depths:
    
        classif = DecisionTreeClassifier(max_depth = max_depth, random_state=42, criterion='gini')
        classif.fit(X_train, y_train)
    
        # Predecir y evaluar sobre el set de entrenamiento
        y_train_pred = classif.predict(X_train)
        precision_train = metrics.precision_score(y_train, y_train_pred)
    
        # Predecir y evaluar sobre el set de evaluación
        y_test_pred = classif.predict(X_test)
        precision_test = metrics.precision_score(y_test, y_test_pred)
    
        #Agrego la informacion a las listas
        train_errors.append(precision_train)
        test_errors.append(precision_test)
        
    return train_errors, test_errors  

max_depths = np.arange(1,30)

train_errors, test_errors = arbol_decisiones(max_depths, X_train, y_train, X_test, y_test)

# Dibujaremos un gráfico con la lista de errores que nos proporciona
# nuestra maravillosa función de creación de arboles.
plt.figure(figsize = (16,8))
plt.plot(max_depths, train_errors,'-x',label='Entrenamiento' )
plt.plot(max_depths,test_errors,'-o',label='Testeo')
plt.legend()
plt.xlabel('Profundidad del árbol de decisiones')
plt.ylabel('Precision (exactitud) de nuestro árbol de decisiones')
plt.show() 

# Emplearemos cross_val_score junto con make_scorer ya importados
classif = DecisionTreeClassifier(random_state = 0, max_depth=4)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 4: ',precision_train.mean())
print('Precision Test depth 4: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=5)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 5: ',precision_train.mean())
print('Precision Test depth 5: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=10)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 12: ',precision_train.mean())
print('Precision Test depth 12: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 13: ',precision_train.mean())
print('Precision Test depth 13: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=14)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 14: ',precision_train.mean())
print('Precision Test depth 14: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=15)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 16: ',precision_train.mean())
print('Precision Test depth 16: ',precision_test.mean(), '\n')

classif = DecisionTreeClassifier(random_state = 0, max_depth=20)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
print('Precision Train depth 17: ',precision_train.mean())
print('Precision Test depth 17: ',precision_test.mean(), '\n')

# Ejecutaremos primero nuestro modelo con 12 profundidades para 
# echar un primer vistazo.
classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)

# Pasaremos a imprimir por pantalla el peso de los atributos
peso = classif.feature_importances_
columna = df_wine.columns.values
feat_import = {'Atributos': columna, 'Peso': peso}
df2 = pd.DataFrame(data=feat_import)
print(df2.sort_values(by = 'Peso', ascending = False))

# Ejecutaremos el árbol de decisiones de 13 profundidades
classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)

propiedades =(df_wine.columns).tolist()
str_propiedades = [str(int) for int in propiedades]
target2 = target.unique().tolist()
str_target = [str(int) for int in target2]
tree_representation = export_text(classif, feature_names = propiedades)
print(tree_representation)

fig = plt.figure(figsize=(50,40))
_ = tree.plot_tree(classif, 
                   feature_names=str_propiedades,  
                   class_names=str_target,
                   filled=True,
                  max_depth = 12)
plt.show()

# Elaboramos una matriz de confusión empleando un mapa de calor
# Para la elección de una profundidad de 5
classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
cmatrix = confusion_matrix(y_test, y_prediccion)
sns.heatmap(cmatrix, annot=True, fmt="d", cmap='seismic', square=True)
plt.title('Matriz de confusión del set de prueba', fontsize = 15)
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.show()

classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
report = classification_report(y_test, y_prediccion)
print('Para una profundidad de 12 \n', report)

classif = DecisionTreeClassifier(random_state = 0, max_depth=12)
classif.fit(X_train, y_train)
y_prediccion = classif.predict(X_test)
precision_train = cross_val_score(classif, X_train, y_train, scoring = 'precision', cv = 5)
precision_test = cross_val_score(classif, X_test, y_test, scoring = 'precision', cv = 5)
fpr, tpr, thresholds = roc_curve(y_test, y_prediccion)
roc_aux = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='r', lw=lw, label='ROC curve (area = %0.2f)' % roc_aux)
plt.plot([0, 1], [0, 1], color='k', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC para el cálculo de AUC para una profundidad de 13')
plt.legend(loc="lower right")
plt.show()

# Randomizamos
rand_forest = RandomForestClassifier(random_state=42)

# Creamos un modelo con RandomizedSearchCV que votará por nosotros.
# Emplearemos por ejemplo el f1-score para realizar la votación
modelos = RandomizedSearchCV(rand_forest,param_distributions={
    'max_depth': np.arange(2, 20)}, cv=5, refit=True, scoring = 'precision')

modelos.fit(X_train, y_train)

print("Los parámetros más TOP: "+str(modelos.best_params_))
print("La exactitud: "+str(modelos.best_score_)+'\n')

#Hago las predicciones con el modelo 
y_train_pred = modelos.predict(X_train)
y_test_pred = modelos.predict(X_test)

report = classification_report(y_test, y_test_pred)
print(report)

data = pd.read_csv('data/df_wine_cleaned.csv')

# Transformamos de nuevo la variable target en 0, 1, 2, 3
data.loc[data.rating == 'D', 'quality'] = 0
data.loc[data.rating == 'C', 'quality'] = 1
data.loc[data.rating == 'B', 'quality'] = 2
data.loc[data.rating == 'A', 'quality'] = 3


data_minoritaria = data[data.Type != 1]
data_mayoritaria = data[data.Type == 1]

data_minoritaria_resample = resample(data_minoritaria, 
                                 replace=True,     
                                 n_samples=700,   
                                 random_state=123) 

data_resample = pd.concat([data_mayoritaria, data_minoritaria_resample])

# Seleccionamos nuestra variable objetivo
y_resample = data_resample['Type']

# Eliminamos las variables que no nos sirven
X_resample = data_resample.drop(['Type', 'rating'], axis=1)

X_train_resample, X_test_resample, y_train_resample, y_test_resample = train_test_split(
    X_resample, y_resample, test_size=0.30, random_state = 42, stratify = y_resample)

y_train_resample = y_train_resample.astype(int)
y_test_resample = y_test_resample.astype(int)

clf = StandardScaler()
X_train_resample = clf.fit_transform(X_train_resample)
X_test_resample = clf.transform(X_test_resample)

max_depths = np.arange(1,30)

train_errors, test_errors = arbol_decisiones(
    max_depths, X_train_resample, y_train_resample,
    X_test_resample, y_test_resample)

# Dibujaremos un gráfico con la lista de errores que nos proporciona
# nuestra maravillosa función de creación de arboles.
plt.figure(figsize = (16,8))
plt.plot(max_depths, train_errors,'-x',label='Entrenamiento' )
plt.plot(max_depths,test_errors,'-o',label='Testeo')
plt.legend()
plt.xlabel('Profundidad del árbol de decisiones')
plt.ylabel('Precision (exactitud) de nuestro árbol de decisiones')
plt.show() 

# Emplearemos cross_val_score junto con make_scorer ya importados
for i in range(2, 21):
    classif = DecisionTreeClassifier(random_state = 0, max_depth=i)
    classif.fit(X_train_resample, y_train_resample)
    y_prediccion = classif.predict(X_test)
    precision_train = cross_val_score(classif, X_train_resample, y_train_resample,
                                      scoring = 'precision', cv = 5)
    precision_test = cross_val_score(classif, X_test_resample, y_test_resample,
                                     scoring = 'precision', cv = 5)
    print('Precision Train depth ', i, ': ',precision_train.mean())
    print('Precision Test depth ', i, ': ',precision_test.mean(), '\n')
    
# Randomizamos
rand_forest = RandomForestClassifier(random_state=42)

# Creamos un modelo con RandomizedSearchCV que votará por nosotros.
# Emplearemos por ejemplo el f1-score para realizar la votación
modelos = RandomizedSearchCV(rand_forest,param_distributions={
    'max_depth': np.arange(2, 20)}, cv=5, refit=True, scoring = 'precision')

modelos.fit(X_train_resample, y_train_resample)

print("Los parámetros más TOP: "+str(modelos.best_params_))
print("La exactitud: "+str(modelos.best_score_)+'\n')

#Hago las predicciones con el modelo 
y_train_pred = modelos.predict(X_train_resample)
y_test_pred = modelos.predict(X_test_resample)

report = classification_report(y_test_resample, y_test_pred)
print(report)

cmatrix = confusion_matrix(y_test_resample, y_test_pred)
sns.heatmap(cmatrix, annot=True, fmt="d", cmap='seismic', square=True)
plt.title('Matriz de confusión del set de prueba', fontsize = 15)
plt.xlabel("Prediccion")
plt.ylabel("Real")
plt.show()