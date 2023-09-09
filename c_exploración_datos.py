import matplotlib.pyplot as plt
import pandas as pd ### para manejo de datos
import sqlite3 as sql


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
