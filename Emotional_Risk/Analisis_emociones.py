# analisis_emociones.py
# Análisis del impacto de las emociones en la toma de riesgo financiero
# Comparación entre inversores comunes (Control) e inversores tratados (Tratamiento)

# Importar librerías
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import ttest_ind

# Estilo de gráficos
sns.set(style="whitegrid")

# =================== CARGA Y LIMPIEZA DE DATOS ===================

# Cargar datos
Common_investors = pd.read_csv('C_round_1.csv')
Treated_investors = pd.read_csv('T_round_1.csv')

# Limpiar columnas innecesarias
for df in [Common_investors, Treated_investors]:
    df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')

# Manejo de valores nulos
emotion_cols = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
for df in [Common_investors, Treated_investors]:
    df.dropna(subset=emotion_cols + ['Numero transazioni'], inplace=True)

print("Datos cargados y limpios")
print("Common investors:", len(Common_investors), "filas")
print("Treated investors:", len(Treated_investors), "filas")

# =================== ANÁLISIS DE CORRELACIÓN ===================

def plot_emotion_analysis(df, emotion_cols, group_name):
    """Gráfica correlación entre emociones y transacciones."""
    corr_series = df[emotion_cols + ['Numero transazioni']].corr()['Numero transazioni'][emotion_cols]
    print(f"\n Correlación en {group_name}:")
    print(corr_series.round(3))

    # Gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr_series.index, y=corr_series.values, palette='viridis')
    plt.title(f'Correlación: Emociones vs. Transacciones ({group_name})')
    plt.xlabel('Emociones')
    plt.ylabel('Coeficiente de correlación')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/correlation_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

    # Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df[emotion_cols + ['Numero transazioni']].corr(), annot=True, cmap='coolwarm', fmt='.2f', center=0)
    plt.title(f'Matriz de correlación ({group_name})')
    plt.tight_layout()
    plt.savefig(f'results/heatmap_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

# Aplicar a ambos grupos
plot_emotion_analysis(Common_investors, emotion_cols, "Common investors")
plot_emotion_analysis(Treated_investors, emotion_cols, "Treated investors")

# =================== BOX PLOT POR NIVEL DE MIEDO ===================

def plot_fear_boxplot(df, fear_col, group_name, palette):
    df[fear_col] = pd.cut(df['fear'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=fear_col, y='Numero transazioni', palette=palette)
    plt.title(f'Transacciones por nivel de miedo ({group_name})')
    plt.ylabel('Número de transacciones')
    plt.tight_layout()
    plt.savefig(f'results/boxplot_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

plot_fear_boxplot(Common_investors.copy(), 'fear_level_C', 'Common investors', 'Blues')
plot_fear_boxplot(Treated_investors.copy(), 'fear_level_T', 'Treated investors', 'Oranges')

# =================== REGRESIÓN LINEAL CON COMPARACIÓN ENTRE GRUPOS ===================

# Lista para almacenar resultados
regression_results = []

def linear_regression_emotions(df, emotion_cols, group_name):
    X = df[emotion_cols]
    y = df['Numero transazioni']
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)
    
    # Guardar coeficientes
    coef_dict = {emo: coef for emo, coef in zip(emotion_cols, model.coef_)}
    coef_dict['Intercept'] = model.intercept_
    coef_dict['R2'] = r2
    coef_dict['Group'] = group_name
    regression_results.append(coef_dict)
    
    # Imprimir resultados
    print(f"\n R² del modelo ({group_name}): {r2:.3f}")
    print("\nCoeficientes (impacto de cada emoción):")
    for emo in emotion_cols:
        print(f"{emo}: {coef_dict[emo]:.3f}")
    
    max_emo = max(emotion_cols, key=lambda x: coef_dict[x])
    min_emo = min(emotion_cols, key=lambda x: coef_dict[x])
    print(f"\n En {group_name}, la emoción que más aumenta transacciones es '{max_emo}' ({coef_dict[max_emo]:.3f})")
    print(f" La que más reduce es '{min_emo}' ({coef_dict[min_emo]:.3f})")

    # Gráfico
    plt.figure(figsize=(10, 6))
    sns.barplot(x=emotion_cols, y=[coef_dict[emo] for emo in emotion_cols], palette='coolwarm')
    plt.title(f'Impacto de emociones en transacciones ({group_name})')
    plt.ylabel('Coeficiente')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'results/regression_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

# Ejecutar regresión
linear_regression_emotions(Common_investors, emotion_cols, "Common investors")
linear_regression_emotions(Treated_investors, emotion_cols, "Treated investors")

# =================== COMPARACIÓN ENTRE GRUPOS (VALOR AGREGADO) ===================

# Crear DataFrame de comparación
coef_df = pd.DataFrame(regression_results).set_index('Group')[emotion_cols].T
print("\n COMPARACIÓN DE COEFICIENTES (Control vs. Tratado):")
print(coef_df.round(3))

# Gráfico comparativo
plt.figure(figsize=(10, 6))
coef_df.plot(kind='bar', ax=plt.gca(), color=['skyblue', 'salmon'], edgecolor='black')
plt.title('Impacto de emociones en transacciones: Control vs. Tratado')
plt.ylabel('Coeficiente de regresión')
plt.xlabel('Emociones')
plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(rotation=45)
plt.legend(title='Grupo')
plt.tight_layout()
plt.savefig('results/coef_comparison.png', dpi=300)
plt.show()

# =================== T-TEST Y ANÁLISIS ADICIONAL ===================

def fear_ttest(df, group_name):
    median_fear = df['fear'].median()
    high_fear = df[df['fear'] > median_fear]['Numero transazioni']
    low_fear = df[df['fear'] <= median_fear]['Numero transacciones']
    t_stat, p_value = ttest_ind(high_fear, low_fear)
    print(f" {group_name} - t-statistic: {t_stat:.3f}, p-value: {p_value:.3f}")
    if p_value < 0.05:
        print(f" Diferencia significativa en {group_name}")
    else:
        print(f" No hay diferencia significativa en {group_name}")

fear_ttest(Common_investors, "Common investors")
fear_ttest(Treated_investors, "Treated investors")

# Scatter plots
def plot_fear_scatter(df, color, group_name):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='fear', y='Numero transazioni', color=color)
    plt.title(f'Relación miedo vs. transacciones ({group_name})')
    plt.xlabel('Nivel de miedo')
    plt.ylabel('Número de transacciones')
    plt.tight_layout()
    plt.savefig(f'results/scatter_{group_name.replace(" ", "_").lower()}.png', dpi=300)
    plt.show()

plot_fear_scatter(Common_investors, 'blue', 'Common investors')
plot_fear_scatter(Treated_investors, 'orange', 'Treated investors')

# =================== COMBINACIÓN DE GRUPOS ===================

Common_investors['grupo'] = 'Control'
Treated_investors['grupo'] = 'Tratamiento'
df_combined = pd.concat([Common_investors, Treated_investors], ignore_index=True)

plt.figure(figsize=(8, 5))
sns.boxplot(data=df_combined, x='grupo', y='fear', palette='Set2')
plt.title('Nivel de miedo: Control vs. Tratamiento')
plt.ylabel('Nivel de miedo')
plt.tight_layout()
plt.savefig('results/fear_by_group.png', dpi=300)
plt.show()

sns.lmplot(data=df_combined, x='fear', y='Numero transazioni', hue='grupo', aspect=1.5)
plt.title('Relación miedo-transacciones por grupo')
plt.xlabel('Nivel de miedo')
plt.ylabel('Número de transacciones')
plt.tight_layout()
plt.savefig('results/lmplot_comparison.png', dpi=300)
plt.show()

print("\n Análisis completado. Resultados guardados en /results")