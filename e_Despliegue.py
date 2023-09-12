
import pandas as pd ### para manejo de datos
import sqlite3 as sql
import joblib
import openpyxl ## para exportar a excel
import numpy as np


###### el despliegue consiste en dejar todo el código listo para una ejecucion automática en el periodo definido:
###### en este caso se ejecutara el proceso de entrenamiento y prediccion anualmente.
if __name__=="__main__":


  ### conectarse a la base de datos ###
  conn=sql.connect("db_empleados")
  cur=conn.cursor()

  ### Ejecutar sql de preprocesamiento inicial y juntarlo
  #### con base de preprocesamiento con la que se entrenó para evitar perdida de variables por conversión a dummies

  funciones.ejecutar_sql('preprocesamiento.sql',cur) ### con las fechas actualizadas explicativas 2023- predecir 2024
  df=pd.read_sql('''select  * from base_nueva''',conn)
  df = df.drop(['Over18','EmployeeCount','index'], axis = 1) #eliminación de variables de una sola categoria

  ####Otras transformaciones en python (imputación, dummies y seleccion de variables)
  df_t= preparar_datos(df)


  ##Cargar modelo y predecir
  bestModel_rfc = joblib.load("bestModel_rfc.pkl")
  predicciones=bestModel_rfc.predict(df_t)
  pd_pred=pd.DataFrame(predicciones, columns=['pred_2017'])


  ###Crear base con predicciones ####

  perf_pred=pd.concat([df['EmployeeID'],df_t,pd_pred],axis=1)

  ####LLevar a BD para despliegue
  perf_pred.loc[:,['EmployeeID', 'pred_2017']].to_sql("perf_pred",conn,if_exists="replace") ## llevar predicciones a BD con ID Empleados


  ####ver_predicciones_bajas ###
  emp_pred_bajo=perf_pred.sort_values(by=["pred_2017"],ascending=True).head(10)

  emp_pred_bajo.set_index('EmployeeID', inplace=True)
  pred=emp_pred_bajo.T

  importancia1 = bestModel_rfc.feature_importances_   #importancia de las variables seleccionadas
  importancia1 = pd.DataFrame(importancia1, columns=['Importancia'])
  X3 = pd.DataFrame(X2.columns, columns=['Variables'])
  # Concatenar las Series al DataFrame X2
  variables_con_importancias = pd.concat([X3, importancia1], axis=1)
  variables_con_importancias.to_excel("variables_con_importancias.xlsx") ### exportar importancia de variables para análisis personalizado

