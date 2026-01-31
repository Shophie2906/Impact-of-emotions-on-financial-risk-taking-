# Investigación en finanzas del comportamiento (Based on PhD Tesis Sofia Poggi, machine learning & facial recognition experimental research)
# By Elena Sánchez, student of Economics and Mathematics
# Universidad Metropolitana, Venezuela
# Date: 2025-08-07

# Este script realiza un análisis exploratorio de los datos de la Round T & Round C


# Preguntas de investigación:
# ¿Las emociones influyen en la toma de decisiones financieras?
# ¿El miedo inducido reduce la actividad de trading?
# ¿Qué emoción tiene el mayor impacto en la aversión al riesgo?
# ¿Existen perfiles emocionales asociados a mayor o menor riesgo?

# Análisis de datos de la Round T & Round C
# Importar librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Cargar datos
Common_investors = pd.read_csv('C_round_1.csv')
Treated_investors = pd.read_csv('T_round_1.csv')

# Limpieza de datos
# Ver columnas
print("Columnas:", Common_investors.columns.tolist())
# Eliminar 'Unnamed: 0' si es solo un índice
for df in [Common_investors, Treated_investors]:
    if 'Unnamed: 0' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
# Ver primeras filas
print(Common_investors.head())
print(Common_investors.info())
print(Common_investors.describe())
print(Common_investors.isnull().sum())
print("Columnas disponibles:")
print(Common_investors.columns.tolist())

print(Treated_investors.head())
print(Treated_investors.info())
print(Treated_investors.describe())
print(Treated_investors.isnull().sum())
print("Columnas disponibles:")
print(Treated_investors.columns.tolist())

# Análisis exploratorio
# Etiquetas
Common_investors['grupo'] = 'Control'
Treated_investors['grupo'] = 'Tratamiento'
# Combined datasets
df_combined = pd.concat([Common_investors, Treated_investors], ignore_index=True)
print(df_combined['grupo'].value_counts())
# Análisis
plt.figure(figsize=(8, 5))
sns.boxplot(data=df_combined, x='grupo', y='Numero transazioni', palette='Set2')
plt.title('Número de transacciones: Control vs. Tratamiento')
plt.ylabel('Transacciones')
plt.show()

# Estadísticas
mean_control = Common_investors['Numero transazioni'].mean()
mean_treatment = Treated_investors['Numero transazioni'].mean()
print(f"Control: {mean_control:.2f}")
print(f"Tratamiento: {mean_treatment:.2f}")
print(f"Diferencia: {mean_control - mean_treatment:.2f}")