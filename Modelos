#### Importar librerias necesarias  ###

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error, f1_score
import sqlite3 as sql
from sklearn.linear_model import LogisticRegression
from sklearn import svm, tree, metrics
from sklearn.ensemble import RandomForestClassifier ##Ensamble con bagging
from sklearn.ensemble import GradientBoostingClassifier ###Ensamble boosting
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import GridSearchCV
import joblib  ### para guardar modelos
import a_funciones as funciones
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import openpyxl
%matplotlib inline

#### Variables dummis

df3=df2.copy()
list_dummies=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']
df_dummies=pd.get_dummies(df3,columns=list_dummies)

y=df_dummies.Attrition
X1= df_dummies.loc[:,~df_dummies.columns.isin(['Attrition'])]
scaler=StandardScaler()
scaler.fit(X1)
X2=scaler.transform(X1)
X=pd.DataFrame(X2,columns=X1.columns)

##### creación de modelos 

m_log = LogisticRegression(max_iter=1000,random_state=42)
m_rf= RandomForestClassifier(random_state=42)
m_tc = tree.DecisionTreeClassifier(random_state=42)
m_gbt=GradientBoostingClassifier(random_state=42)


modelos=list([m_log, m_rf, m_tc, m_gbt])

### selección de variables 

var_names= funciones.sel_variables(modelos,X,y,threshold="2.35*mean")
var_names.shape

modelo=modelos[0]
modelo.fit(X,y)

X2=X[var_names] ### matriz con variables seleccionadas
X2.info()

acc_df = funciones.medir_modelos(modelos,"f1",X,y,2) ## base con todas las variables
acc_varsel= funciones.medir_modelos(modelos,"f1",X2,y,2) ### base con variables seleccionadas

acc=pd.concat([acc_df,acc_varsel],axis=1)
acc.columns=['log', 'rf', 'tc','gbt',
       'log_sel','rf_sel', 'tc_sel','gb_sel']

acc_df.plot(kind='box') #### gráfico para modelos todas las varibles
acc_varsel.plot(kind='box') ### gráfico para modelo variables seleccionadas
acc.plot(kind='box') ### gráfico para modelos sel y todas las variables

##### afinamiento de hiperparametros 

# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

# Definición de cuadricula de hiperparametros
parameters = {'n_estimators': [20,40,80,100,150],
              'max_depth':[10,12,14,15,16],
              'max_leaf_nodes': [50,70,100, 300]}

#Definición del modelo
rfc = RandomForestClassifier(random_state=42, class_weight='balanced')

grid_search = GridSearchCV(rfc, parameters, cv=5, scoring='f1', n_jobs=-1)
grid_result = grid_search.fit(X_train, y_train)

print('Best Params: ', grid_result.best_params_)
print('Best Score: ', grid_result.best_score_)

bestModel_rfc=grid_result.best_estimator_
print("F1:{:.2f}".format(bestModel_rfc.score(X_test,y_test)))

# ==============================================================================
#### Entrenamiento de los modelos

ranfor = RandomForestClassifier(
            n_estimators = 150,
            criterion    = 'gini',
            max_depth    = 16,
            max_leaf_nodes = 300,
            max_features = None,
            oob_score    = False,
            n_jobs       = -1,
            random_state = 123
         )
ranfor.fit(X_train, y_train)

ranfor.feature_importances_
X2.columns

# Métricas de desempeño
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor.predict(X_test)))

y_hat=ranfor.predict(X_test)
# Matriz de confusión
fig = plt.figure(figsize=(11,11))
cm = confusion_matrix(y_test,y_hat, labels=ranfor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=ranfor.classes_)
disp.plot(cmap='gist_earth')
plt.show()












