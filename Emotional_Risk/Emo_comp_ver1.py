
# Preguntas de investigación:
# ¿Las emociones influyen en la toma de decisiones financieras?
# ¿El miedo inducido reduce la actividad de trading?
# ¿Qué emoción tiene el mayor impacto en la aversión al riesgo?
# ¿Existen perfiles emocionales asociados a mayor o menor riesgo?

# Análisis de datos de emociones
# Importar librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
Common_investors = pd.read_csv('C_round_1.csv')
Treated_investors = pd.read_csv('T_round_1.csv')

# Limpieza de datos
# Ver columnas
print(Common_investors.head())
print(Treated_investors.head())
#Limpiar 'Unnamed: 0' si es solo un índice
for df in [Common_investors, Treated_investors]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)  
# Manejo valores nulos 
dropna_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'Numero transazioni']
Common_investors.dropna(subset=dropna_cols, inplace=True)
Treated_investors.dropna(subset=dropna_cols, inplace=True)
print("Valores nulos en Common investors:")
print(Common_investors.isnull().sum())
print("Valores nulos en Treated investors:")
print(Treated_investors.isnull().sum())


# ¿Qué emoción tiene el mayor impacto en la aversión al riesgo?

# Lineal correlations between emotions and Number of transactions
# Common investors
# Emociones C
emotion_cols_C = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
# Calcular correlaciones lineales C
correlations = Common_investors[emotion_cols_C + ['Numero transazioni']].corr()['Numero transazioni'][emotion_cols_C]
print("Correlación entre emociones y transacciones en Common investors:")
print(correlations.round(3))
# Grafico Bars C
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations.index, y=correlations.values, palette='viridis')
plt.title('Correlación entre emociones y número de transacciones (Common investors)')
plt.xlabel('Emociones')
plt.ylabel('Correlación')
plt.xticks(rotation=45)
plt.show()
# Grafico Heatmap C
plt.figure(figsize=(10, 6))
sns.heatmap(Common_investors[emotion_cols_C + ['Numero transazioni']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación de emociones y número de transacciones (Common investors)')
plt.show()  

# Treated investors
# Emociones T
emotion_cols_T = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
# Calcular correlaciones lineales T
correlations_T = Treated_investors[emotion_cols_T + ['Numero transazioni']].corr()['Numero transazioni'][emotion_cols_T]
print("Correlación entre emociones y transacciones en Treated investors:")
print(correlations_T.round(3))
# Grafico Bars T
plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_T.index, y=correlations_T.values, palette='viridis')
plt.title('Correlación entre emociones y número de transacciones (Treated investors)')
plt.xlabel('Emociones')
plt.ylabel('Correlación')
plt.xticks(rotation=45)
plt.show()
# Grafico Heatmap T
plt.figure(figsize=(10, 6))
sns.heatmap(Treated_investors[emotion_cols_T + ['Numero transazioni']].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlación de emociones y número de transacciones (Treated investors)')
plt.show()

# Tendencias y grupos FEAR
# Categorizar miedo
Common_investors['fear_level_C'] = pd.cut(Common_investors['fear'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
Treated_investors['fear_level_T'] = pd.cut(Treated_investors['fear'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
# Boxplot C
sns.boxplot(data=Common_investors, x='fear_level_C', y='Numero transazioni', palette='Blues')
plt.title('Transacciones por nivel de miedo (Common investors)')
plt.ylabel('Número de transacciones')
plt.show()
# Boxplot T
sns.boxplot(data=Treated_investors, x='fear_level_T', y='Numero transazioni', palette='Oranges')
plt.title('Transacciones por nivel de miedo (Treated investors)')
plt.ylabel('Número de transacciones')
plt.show()



# Regresión lineal 
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# Regresión lineal C
# Variables C
X = Common_investors[emotion_cols_C]  # emociones
y = Common_investors['Numero transazioni']  # transacciones
# Modelo
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
# Resultados
print("R² del modelo (Common investors):", round(r2_score(y, y_pred), 3))
print("\nCoeficientes (impacto de cada emoción en Common investors):")
for emo, coef in zip(emotion_cols_C, model.coef_):
    print(f"{emo}: {coef:.3f}")
# Regresión lineal T
# Variables T
X_T = Treated_investors[emotion_cols_T]  # emociones
y_T = Treated_investors['Numero transazioni']  # transacciones
# Modelo T 
model_T = LinearRegression()
model_T.fit(X_T, y_T)
y_pred_T = model_T.predict(X_T)
# Resultados T
print("R² del modelo (Treated investors):", round(r2_score(y_T, y_pred_T), 3))
print("\nCoeficientes (impacto de cada emoción en Treated investors):")
for emo, coef in zip(emotion_cols_T, model_T.coef_):
    print(f"{emo}: {coef:.3f}")
    

# Prueba de hipótesis: ¿el miedo reduce significativamente las transacciones?
# Un t-test para comparar participantes con alto vs. bajo miedo.
from scipy.stats import ttest_ind
# Dividir por miedo C
high_fear = Common_investors[Common_investors['fear'] > Common_investors['fear'].median()]['Numero transazioni']
low_fear = Common_investors[Common_investors['fear'] <= Common_investors['fear'].median()]['Numero transazioni']
# Prueba t C
t_stat, p_value = ttest_ind(high_fear, low_fear)
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.3f}")
if p_value < 0.05:
    print("✅ Diferencia estadísticamente significativa: el miedo reduce transacciones (Common investors)")
else:
    print("❌ No hay diferencia significativa (Common investors)")
#Dividir por miedo T
high_fear_T = Treated_investors[Treated_investors['fear'] > Treated_investors['fear'].median()]['Numero transazioni']
low_fear_T = Treated_investors[Treated_investors['fear'] <= Treated_investors['fear'].median()]['Numero transazioni']
# Prueba t T
t_stat_T, p_value_T = ttest_ind(high_fear_T, low_fear_T)
print(f"t-statistic (Treated investors): {t_stat_T:.3f}")
print(f"p-value (Treated investors): {p_value_T:.3f}")
if p_value_T < 0.05:    
    print("✅ Diferencia estadísticamente significativa: el miedo reduce transacciones (Treated investors)")
else:
    print("❌ No hay diferencia significativa (Treated investors)")


# ¿El miedo inducido reduce la actividad de trading?

# Análisis de la relación entre miedo y número de transacciones
# Relación miedo y transacciones a Treated Investors
plt.figure(figsize=(8,5))
sns.scatterplot(data=Treated_investors, x='fear', y='Numero transazioni', color='orange')
plt.title('Relación entre miedo y número de transacciones (Treated investors)')
plt.xlabel('Nivel de miedo')
plt.ylabel('Número de transacciones')
plt.show()
#Relacion miedo y transacciones a Common Investors
plt.figure(figsize=(8, 5))
sns.scatterplot(data=Common_investors, x='fear', y='Numero transazioni', color='blue')
plt.title('Relación entre miedo y número de transacciones (Common investors)')
plt.xlabel('Nivel de miedo')
plt.ylabel('Número de transacciones')
plt.show()  

# Comparación miedo entre grupos
# Etiquetas
Common_investors['grupo'] = 'Control'
Treated_investors['grupo'] = 'Tratamiento'
# Combined datasets
df_combined = pd.concat([Common_investors, Treated_investors], ignore_index=True)
print(df_combined['grupo'].value_counts())
# Comparación miedo entre grupos
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_combined, x='grupo', y='fear', palette='Set2')
plt.title('Nivel de miedo: Control vs. Tratamiento')
plt.ylabel('Nivel de miedo')
plt.show()  
# Comparación miedo y transacciones entre grupos
plt.figure(figsize=(8, 5))
sns.lmplot(data=df_combined, x='fear', y='Numero transazioni', hue='grupo', aspect=1.5)
plt.title('Relación entre miedo y número de transacciones por grupo')
plt.xlabel('Nivel de miedo')
plt.ylabel('Número de transacciones')
plt.show()  





