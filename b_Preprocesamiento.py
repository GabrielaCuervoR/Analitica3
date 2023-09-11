from platform import python_version ## versión de python
import pandas as pd ### para manejo de datos
import sqlite3 as sql #### para bases de datos sql
import a_funciones as funciones  ###archivo de funciones propias
import sys ## saber ruta de la que carga paquetes
from sklearn.preprocessing import LabelEncoder ### Transformar la variable objetivo en numérica.

##Se cargan las bases de datos almacenadas en el github
df_data= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/general_data.csv'
df_retiros= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/retirement_info.csv'
df_employee= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/employee_survey_data.csv'
df_manager= 'https://raw.githubusercontent.com/GabrielaCuervoR/Analitica3/main/Data/manager_survey_data.csv'

##Se leen la base de datos con la información arreglada
df_data=pd.read_csv(df_data, sep= ';' )
df_retiros=pd.read_csv(df_retiros, sep=';')
df_employee=pd.read_csv(df_employee)
df_manager=pd.read_csv(df_manager)

### resumen información tablas

df_data.info()
df_retiros.info()
df_employee.info()
df_manager.info()

#### crear base de datos para manejo de datos ####

conn= sql.connect("db_empleados") ## Con este comando se crea la base de datos o si ya existe se conecta

##Se lleva cada base de datos a sql
df_employee.to_sql("employee_survey_data",conn,if_exists="replace")
df_data.to_sql("general_data",conn,if_exists="replace")
df_manager.to_sql("manager_survey_data",conn,if_exists="replace")
df_retiros.to_sql("retirement_info",conn,if_exists="replace")

##Se ejecuta el archivo de preprocesamiento de sql
cur= conn.cursor()
funciones.ejecutar_sql('preprocesamiento.sql', cur)

## Se lee la tabla de datos y se almacena en un dataframe
df2=pd.read_sql("select * from base_nueva", conn)
df2 = df2.drop(columns=['index'], axis=1)## Se elimina la columna index que por defecto la sube sql 

### Se analiza la información de la base de datos

df2.info()

### Se verifican las categorias y obsevaciones de cada variable categorica, para ver si se deben agrupar por si tiene muchas categorias.

pd.read_sql("""select BusinessTravel,count(*) 
                            from base_nueva 
                            group by BusinessTravel""", conn)

pd.read_sql("""select Department,count(*) 
                            from base_nueva 
                            group by Department""", conn)

pd.read_sql("""select EducationField ,count(*) 
                            from base_nueva 
                            group by EducationField """, conn)

pd.read_sql("""select Gender ,count(*) 
                            from base_nueva 
                            group by Gender""", conn)

##Esta variable JobRole es la que tiene más categorias pero lo que representa son el cargo de cada trabajador por lo tanto se procede a dejarla asi.
pd.read_sql("""select  JobRole ,count(*) 
                            from base_nueva 
                            group by JobRole """, conn)

pd.read_sql("""select MaritalStatus ,count(*) 
                            from base_nueva 
                            group by MaritalStatus """, conn)

##Esta variable Over18 tiene una sola categoria por lo que no aportaria al modelo y se debe eliminar.
pd.read_sql("""select  Over18 ,count(*) 
                            from base_nueva 
                            group by Over18 """, conn)

##Esta variable en nuestra variable objetivo por lo tanto se debe convertir a numerica para que el modelo pueda predecir mejor.
pd.read_sql("""select  Attrition ,count(*) 
                            from base_nueva 
                            group by Attrition """, conn)
##Se verifica la cantidad de nulos de la base de datos
df2.isnull().sum()

###CONCLUSIONES###
##Se observa cuanto nulos tiene cada variable para más adelante proceder a la imputación (Esto se va a hacer en el archivo donde se crea el modelo).
##El análisis de las variables numéricas se va a realizar en la parte de exploración de datos.
##Se decide que las variables categóricas se van a rellenar con la moda que es el valor que más se repite.
##La variable Attrition se reemplazan los nulos por la palabra No. Ya que nuestra variable objetivo nos va a decir si un empleado se fue o no y también se observa que esta un poco sesgada los datos entonces hay que hacerle una estandarización a la variable.
##Las variables numéricas se van a rellenar con la mediana porque es una medida  que no se ve afectada por valores atípicos de los datos.
