import matplotlib.pyplot as plt
import pandas as pd ### para manejo de datos
import sqlite3 as sql

cur= conn.cursor()
funciones.ejecutar_sql('preprocesamiento.sql', cur)

df2=pd.read_sql("select * from base_nueva", conn)

df2 = df2.drop(columns=['index'], axis=1)##Se elimina la columna index que por defecto la sube sql

##Gráficos variables categóricas
plt.figure(figsize=(20, 10))
plt.subplot(3,3,1)
df2['Attrition'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(3,3,2)
df2['BusinessTravel'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(3,3,3)
df2['Department'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(3,3,7)
df2['EducationField'].value_counts().plot(kind='bar')
plt.subplot(3,3,5)
df2['Gender'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(3,3,8)
df2['JobRole'].value_counts().plot(kind='bar')
plt.subplot(3,3,4)
df2['MaritalStatus'].value_counts().plot(kind='pie',autopct='%.2f')
plt.subplot(3,3,6)
df2['Over18'].value_counts().plot(kind='pie',autopct='%.2f')

 ##Gráficos de las variables numéricas
df2.hist(bins=30, figsize=(20, 15))

df2.info() #Visualizar tipos de datos de cada variable

df2.describe()

##Explorar variable objetivo
df2['Attrition'] = df2['Attrition'].fillna('No') #se rellenan los nulos de la variable objetivo por No

#Histograma de la variable objetivo
df2['Attrition'].value_counts().plot(kind='bar')

y = df2.Attrition ##Variable Objetivo
# LabelEncoder: Transformar la variable objetivo en numérica.
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(y[0:5])

print(le.classes_)

df2['Attrition'] = y

#se hizo la correlación de la variable ojetivo con las numéricas
corr = df2.corr()
corr[['Attrition']].sort_values(by = 'Attrition',ascending = False)\
.style.background_gradient()
