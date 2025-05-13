import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

# 1. Cargar el archivo CSV
dataset_path = "data/StudentExamSuccess.csv"
df = pd.read_csv(dataset_path)

# 2. Seleccionar las DOS variables
cols = ["prev_grade", "attendance_rate"]
X = df[cols].values
y = df["pass_exam"].values

# 3. Separar train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0, stratify=y
)

# 4. Escalado
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 5. Entrenar modelo
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# 6. Graficar frontera de decisión

# Datos des-escalados solo para visualización
X_vis, y_vis = sc.inverse_transform(X_test), y_test

# Mallado
grade_min, grade_max = X_vis[:, 0].min() - 5, X_vis[:, 0].max() + 5
att_min, att_max = X_vis[:, 1].min() - 0.05, X_vis[:, 1].max() + 0.05

X1, X2 = np.meshgrid(
    np.arange(start=grade_min, stop=grade_max, step=0.25),
    np.arange(start=att_min, stop=att_max, step=0.005),
)

# Predicción sobre cada punto del grid
grid_preds = clf.predict(sc.transform(np.c_[X1.ravel(), X2.ravel()])).reshape(X1.shape)

plt.figure(figsize=(8, 6))
plt.contourf(X1, X2, grid_preds, alpha=0.75, cmap=ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Puntos reales del conjunto de prueba
for i, label in enumerate(np.unique(y_vis)):
    plt.scatter(
        X_vis[y_vis == label, 0],
        X_vis[y_vis == label, 1],
        c=ListedColormap(("red", "green"))(i),
        edgecolor="k",
        label=f"Aprobó = {label}",
    )

plt.title("Regresión logística – Conjunto de prueba")
plt.xlabel("Nota previa")
plt.ylabel("Asistencia promedio")
plt.legend()
plt.tight_layout()
plt.show()

# --- ANALISIS ---

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Predicciones y probabilidades
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Métricas clave
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

# Conclusion
print(
    f"""
MODELO DE REGRESIÓN LOGÍSTICA – ANÁLISIS DE RESULTADOS

Para este modelo se han analizado datos de estudiantes universitarios,
con el objetivo de predecir la probabilidad de aprobar un examen final.
Se tomaron como variables independientes la *nota previa* obtenida en cursos anteriores
y el *porcentaje de asistencia promedio* a clases. La variable dependiente es binaria:
1 si el estudiante aprobó el examen, y 0 si lo reprobó.

En el **conjunto de prueba** (25 % de la muestra), se obtuvieron los siguientes resultados:

  • Exactitud (accuracy):        {acc:.3f}
  • Precisión:                   {prec:.3f}
  • Recall (sensibilidad):       {rec:.3f}
  • F1-score:                    {f1:.3f}
  • Área bajo la curva ROC (AUC): {auc:.3f}

La matriz de confusión refleja {cm[1,1]} verdaderos positivos (TP) y {cm[0,0]} verdaderos negativos (TN),
con {cm[0,1]} falsos positivos (FP) y {cm[1,0]} falsos negativos (FN).

El modelo alcanza una precisión general del {acc*100:.1f} %, identificando correctamente
el {rec*100:.1f} % de los casos de aprobación. El AUC de {auc:.2f} respalda una adecuada
capacidad de discriminación entre estudiantes que aprueban y los que no. Por tanto,
las variables seleccionadas son relevantes para explicar el rendimiento académico
en este escenario.
"""
)
