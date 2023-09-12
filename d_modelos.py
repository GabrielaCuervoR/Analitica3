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
from sklearn.preprocessing import LabelEncoder
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

# ==============================================================================
####### 1. Se llama a la base de datos a trabajar
####### 2. Se verifican datos nulos y faltantes
####### 3. Se imputan nulos y se eliminan variables de una sola categoria
####### 4. Se transforma la variable objetivo a numérica
####### 5. Se dejan todas las variables numéricas en formato int
####### 6. Variables dummis
####### 7. Creación de modelos
####### 8. Selección de variables
####### 9. Afinamiento de hiperparámetros mejor modelo
####### 10. Creación de modelo afinado
####### 11. Entramiento del modelo
####### 12. Importancia de las variables
####### 13. Métricas de desempeño
####### 14. Matriz de confusión
# ==============================================================================
#### Se trae la base de datos nueva para iniciar con el modelo ####

conn= sql.connect("db_empleados") ## Con este comando se crea la base de datos o si ya existe se conecta

cur= conn.cursor()
funciones.ejecutar_sql('preprocesamiento.sql', cur)

df2=pd.read_sql("select * from base_nueva", conn)
df2 = df2.drop(columns=['index'], axis=1)##Se elimina la columna index que por defecto la sube sql



### verificación de nulos y datos faltantes
# ==============================================================================
df2.isnull().sum() #Mirar la cantidad de nulos de cada variable



### imputación y eliminación de varibles 
# ==============================================================================
list_cat=['EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance'] #variables categóricas
list_num=['NumCompaniesWorked','TotalWorkingYears'] #variables numéricas

funciones.imputar_fc(df2,list_cat)
funciones.imputar_fn(df2,list_num)
df2 = df2.drop(['Over18','EmployeeCount'], axis = 1) #eliminación de variables de una sola categoria



### verificación de la base datos
# ==============================================================================
df2.isnull().sum() #se verifica que quede completa



### Transformación de variable objetivo
# ==============================================================================
df2['Attrition'] = df2['Attrition'].fillna('No') #se rellenan los nulos de la variable objetivo por No
df2['Attrition'].value_counts() #se verifica el número de cada categória



#### Preparación de los datos 
# ==============================================================================
y = df2.Attrition ##Variable Objetivo 
le = LabelEncoder() 
y = le.fit_transform(y) ##Transformación a númerica

print(y[0:5])  #verificación

print(le.classes_)

df2['Attrition'] = y ## Retornando la variable objetivo ya numérica a la base de datos



#### Cambiar tipo de datos de float a int
# ==============================================================================
df2.EnvironmentSatisfaction = df2.EnvironmentSatisfaction.astype(int)
df2.JobSatisfaction  = df2.JobSatisfaction.astype(int)
df2.WorkLifeBalance = df2.WorkLifeBalance.astype(int)
df2.NumCompaniesWorked = df2.NumCompaniesWorked.astype(int)
df2.TotalWorkingYears = df2.TotalWorkingYears.astype(int)



#### Variables dummis
# ==============================================================================
df3=df2.copy()
list_dummies=['BusinessTravel','Department','EducationField','Gender','JobRole','MaritalStatus']
df_dummies=pd.get_dummies(df3,columns=list_dummies)

y=df_dummies.Attrition
X1= df_dummies.loc[:,~df_dummies.columns.isin(['Attrition','EmployeeID'])] #No se tiene en cuenta la variable objetivo ni el id del empleado
scaler=StandardScaler()  
scaler.fit(X1)  #Estandarización variables
X2=scaler.transform(X1)
X=pd.DataFrame(X2,columns=X1.columns)



##### Creación de modelos 
# ==============================================================================
#Se crean varios modelos para su respectiva comparación de desempeño
m_log = LogisticRegression(max_iter=1000,random_state=42, class_weight='balanced')
m_rf= RandomForestClassifier(random_state=42, class_weight='balanced')
m_tc = tree.DecisionTreeClassifier(random_state=42, class_weight='balanced')
m_gbt=GradientBoostingClassifier(random_state=42)


modelos=list([m_log, m_rf, m_tc, m_gbt])




#### Selección de variables 
# ==============================================================================
## Se seleccionan las variables con más peso en los modelos
var_names= funciones.sel_variables(modelos,X,y,threshold="2.35*mean") #se decide usar el 2.35 mean para tener solo 11 variables en el modelo
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




##### Afinamiento de hiperparametros 
# ==============================================================================
# Separación en conjuntos de entrenamiento y validación con 80% de muestras para entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

# Definición de cuadricula de hiperparametros
parameters = {'n_estimators': [20,40,80,100,150],
              'max_depth':[10,12,14,15,16],
              'max_leaf_nodes': [50,70,100, 300]}




##### Definición del modelo
# ==============================================================================
rfc = RandomForestClassifier(random_state=42, class_weight='balanced') #creación modelo para afinar

grid_search = GridSearchCV(rfc, parameters, cv=5, scoring='f1', n_jobs=-1) #métrica f1 score
grid_result = grid_search.fit(X_train, y_train)

print('Best Params: ', grid_result.best_params_)
print('Best Score: ', grid_result.best_score_)

bestModel_rfc=grid_result.best_estimator_    ### modelo con mejor desempeño
print("F1:{:.2f}".format(bestModel_rfc.score(X_test,y_test)))

# ==============================================================================
#### Entrenamiento del modelo

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




#### Importancia de las variables
# ==============================================================================
importancia = ranfor.feature_importances_   #importancia de las variables seleccionadas
importancia = pd.DataFrame(importancia, columns=['Importancia'])
X3 = pd.DataFrame(X2.columns, columns=['Variables'])

### Unión de las variables con la importancia

X2_con_importancias = pd.concat([X3, importancia], axis=1)
X2_con_importancias.sort_values(by=['Importancia'], ascending=False) #visualización de acuerdo a su nivel de importancia




######## Métricas de desempeño
# ==============================================================================
print ("Train - Accuracy :", metrics.accuracy_score(y_train, ranfor.predict(X_train)))
print ("Train - classification report:\n", metrics.classification_report(y_train, ranfor.predict(X_train)))
print ("Test - Accuracy :", metrics.accuracy_score(y_test, ranfor.predict(X_test)))
print ("Test - classification report :", metrics.classification_report(y_test, ranfor.predict(X_test)))



###### Matriz de confusión
# ==============================================================================
y_hat=ranfor.predict(X_test)
fig = plt.figure(figsize=(11,11))
cm = confusion_matrix(y_test,y_hat, labels=ranfor.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=ranfor.classes_)
disp.plot(cmap='gist_earth')
plt.show()





###### Exportar y guardar objetos 
# ==============================================================================

joblib.dump(rfc, "rfc.pkl") ##  Modelo final con variables seleccionadas
joblib.dump(m_rf, "m_rf.pkl") ## Modelo con todas las variables
joblib.dump(list_cat, "list_cat.pkl") ### para realizar imputacion variables categóricas
joblib.dump(list_num, "list_num.pkl") ### para realizar imputacion variables numéricas
joblib.dump(list_dummies, "list_dummies.pkl")  ### para convertir a dummies
joblib.dump(var_names, "var_names.pkl")  ### para variables con que se entrena modelo
joblib.dump(scaler, "scaler.pkl") ## 


rfc = joblib.load("rfc.pkl")
m_rf = joblib.load("m_rf.pkl")
list_cat=joblib.load("list_cat.pkl")
list_num=joblib.load("list_num.pkl")
list_dummies=joblib.load("list_dummies.pkl")
var_names=joblib.load("var_names.pkl")
scaler=joblib.load("scaler.pkl") 












