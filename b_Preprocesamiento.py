from platform import python_version ## versión de python
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones  ###archivo de funciones propias
import sys ## saber ruta de la que carga paquetes
from sklearn.preprocessing import LabelEncoder ### Transformar la variable objetivo en numérica.

### Carga de base de datos
action='https://raw.githubusercontent.com/juancamiloespana/aplicacionesanalitica/main/data/tbl_Action.csv'

df_data='https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/general_data.csv'
df_retiros= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/retirement_info.csv'
df_employee= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/employee_survey_data.csv'
df_manager= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/manager_survey_data.csv'


df_data=pd.read_csv(df_data)
df_retiros=pd.read_csv(df_retiros)
df_employee=pd.read_csv(df_employee)
df_manager=pd.read_csv(df_manager)

### resumen información tablas

df_data.info()
df_retiros.info()
df_employee.info()
df_manager.info()

#### crear base de datos para manejo de datos ####

conn= sql.connect("db_empleados") ## Con este comando se crea la base de datos o si ya existe se conecta
df2=pd.read_sql("select * from base_nueva", conn)

### verificación de nulos y datos faltantes

df2.isnull().sum() #Mirar la cantidad de nulos de cada variable

### imputación y eliminación de varibles 

list_cat=['EnvironmentSatisfaction','JobSatisfaction','WorkLifeBalance']
list_num=['NumCompaniesWorked','TotalWorkingYears']

a_funciones.imputar_fc(df2,list_cat)
a_funciones.imputar_fn(df2,list_num)
df2 = df2.drop(['Over18','retirementDate', 'retirementType', 'resignationReason','StandardHours','EmployeeID','EmployeeCount'], axis = 1)

### verificación de la base datos

df2.isnull().sum()

### Transformación de variable objetivo

df2['Attrition'] = df2['Attrition'].fillna('No')
df2['Attrition'].value_counts()

#### preparación de los datos 

y = df2.Attrition ##Variable Objetivo
le = LabelEncoder()
y = le.fit_transform(y)

print(y[0:5])

print(le.classes_)

df2['Attrition'] = y

#### Cambiar tipo de datos de float a int

df2.EnvironmentSatisfaction = df2.EnvironmentSatisfaction.astype(int)
df2.JobSatisfaction  = df2.JobSatisfaction.astype(int)
df2.WorkLifeBalance = df2.WorkLifeBalance.astype(int)
df2.NumCompaniesWorked = df2.NumCompaniesWorked.astype(int)
df2.TotalWorkingYears = df2.TotalWorkingYears.astype(int)








df2 = df2.drop(['Over18','retirementDate', 'retirementType', 'resignationReason','StandardHours','EmployeeID','EmployeeCount'], axis = 1)
