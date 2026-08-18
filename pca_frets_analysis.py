# ============================================================
# ANÁLISIS DE COMPONENTES PRINCIPALES - BASE DE DATOS FRETS
# ============================================================
#
# Base de datos: frets
# Paquete original: boot (R)
#
# La base frets contiene medidas de la cabeza de 25 familias.
#
# Variables:
#   l1: longitud de la cabeza del primer hijo (mm)
#   l2: longitud de la cabeza del segundo hijo (mm)
#   b1: anchura de la cabeza del primer hijo (mm)
#   b2: anchura de la cabeza del segundo hijo (mm)
#
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ============================================================
# LIBRERÍAS Y DATOS
# ============================================================

# Datos del dataset frets (de la librería boot de R)
# Medidas de la cabeza de 25 familias
frets_data = {
    'l1': [191, 195, 181, 183, 176, 208, 189, 197, 188, 192, 179, 183, 174, 190, 188, 176, 197, 190, 180, 194, 180, 189, 197, 185, 194],
    'l2': [179, 201, 184, 177, 176, 191, 179, 187, 192, 205, 161, 171, 182, 188, 174, 190, 194, 188, 179, 191, 192, 194, 190, 176, 184],
    'b1': [149, 152, 148, 154, 147, 158, 152, 161, 150, 157, 140, 152, 150, 159, 151, 158, 159, 157, 154, 157, 155, 159, 156, 151, 148],
    'b2': [131, 132, 127, 140, 141, 145, 132, 137, 130, 138, 124, 132, 127, 137, 133, 145, 141, 139, 135, 142, 139, 152, 142, 137, 139]
}

frets = pd.DataFrame(frets_data)

print("=" * 60)
print("COMPROBACIÓN INICIAL DE LOS DATOS")
print("=" * 60)
print("\nPrimeras filas:")
print(frets.head())
print("\nInformación de las variables:")
print(frets.info())
print("\nEstadísticas descriptivas:")
print(frets.describe())


# ============================================================
# EJERCICIO 1
# ACP SOBRE b1 Y b2
# ============================================================

print("\n" + "=" * 60)
print("EJERCICIO 1: ACP SOBRE b1 Y b2")
print("=" * 60)

# Seleccionamos las variables de anchura de la cabeza
head_dat = frets[['b1', 'b2']]

# --------------------------------------------------------
# 1.1 MATRIZ DE COVARIANZAS
# --------------------------------------------------------

print("\n1.1 MATRIZ DE COVARIANZAS")
print("-" * 60)

cov_head = head_dat.cov()
print("\nMatriz de covarianzas:")
print(cov_head)

# Varianzas de cada variable
print("\nVarianzas de cada variable:")
print(np.diag(cov_head.values))

# Varianza total
var_total = np.sum(np.diag(cov_head.values))
print(f"\nVarianza total: {var_total:.6f}")


# --------------------------------------------------------
# 1.2 DESCOMPOSICIÓN ESPECTRAL
# --------------------------------------------------------

print("\n1.2 DESCOMPOSICIÓN ESPECTRAL")
print("-" * 60)

# Valores propios y vectores propios
eigenvalues, eigenvectors = np.linalg.eig(cov_head.values)

# Ordenar por valores propios en orden descendente
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print("\nValores propios:")
print(eigenvalues)

# Comprobación de la varianza total
print(f"\nSuma de valores propios (varianza total): {np.sum(eigenvalues):.6f}")

# Proporción de varianza explicada por la primera CP
prop_var_pc1 = eigenvalues[0] / np.sum(eigenvalues)
print(f"\nProporción de varianza explicada por CP1: {prop_var_pc1:.6f}")


# --------------------------------------------------------
# 1.3 VECTORES PROPIOS
# --------------------------------------------------------

print("\n1.3 VECTORES PROPIOS")
print("-" * 60)

print("\nPrimer vector propio:")
print(eigenvectors[:, 0])

print("\nSegundo vector propio:")
print(eigenvectors[:, 1])


# --------------------------------------------------------
# 1.4 COORDENADAS DE LA PRIMERA OBSERVACIÓN
# --------------------------------------------------------

print("\n1.4 COORDENADAS DE LA PRIMERA OBSERVACIÓN")
print("-" * 60)

# Medias de las variables
means_head = head_dat.mean().values
print(f"\nMedias de las variables:")
print(f"b1: {means_head[0]:.6f}, b2: {means_head[1]:.6f}")

# Valores de la primera familia
obs_1 = head_dat.iloc[0].values
print(f"\nPrimera familia:")
print(f"b1: {obs_1[0]}, b2: {obs_1[1]}")

# Centramos la primera observación
obs_1_centered = obs_1 - means_head

# Coordenada de la primera observación sobre la CP1
score_pc1 = eigenvectors[:, 0] @ obs_1_centered
print(f"\nCoordenada de la primera observación sobre CP1: {score_pc1:.6f}")

# Coordenada de la primera observación sobre la CP2
score_pc2 = eigenvectors[:, 1] @ obs_1_centered
print(f"Coordenada de la primera observación sobre CP2: {score_pc2:.6f}")


# --------------------------------------------------------
# 1.5 REPRESENTACIÓN DE LAS COMPONENTES PRINCIPALES
# --------------------------------------------------------

print("\n1.5 REPRESENTACIÓN DE LAS COMPONENTES PRINCIPALES")
print("-" * 60)

# Pendiente y ordenada de la primera componente
a1 = means_head[1] - (eigenvectors[1, 0] * means_head[1]) / eigenvectors[0, 0]
b1_slope = eigenvectors[1, 0] / eigenvectors[0, 0]

# Pendiente y ordenada de la segunda componente
a2 = means_head[1] - ((-eigenvectors[0, 1] * means_head[1]) / eigenvectors[1, 0])
b2_slope = -eigenvectors[0, 1] / eigenvectors[1, 0]

# Gráfico de las observaciones y las componentes
plt.figure(figsize=(10, 8))
plt.scatter(head_dat['b1'], head_dat['b2'], alpha=0.6, s=100)

# Primera componente principal
x_range = np.linspace(137, 167, 100)
plt.plot(x_range, a1 + b1_slope * x_range, 'r-', label='CP1', linewidth=2)

# Segunda componente principal
plt.plot(x_range, a2 + b2_slope * x_range, 'b--', label='CP2', linewidth=2)

plt.xlabel('Anchura del primer hijo (mm)')
plt.ylabel('Anchura del segundo hijo (mm)')
plt.title('ACP sobre la anchura de la cabeza')
plt.xlim(137, 167)
plt.ylim(130, 159)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('c:\\Users\\usuario\\OneDrive\\Escritorio\\python baby\\pca_ejercicio1.png', dpi=150)
print("\nGráfico guardado como 'pca_ejercicio1.png'")
plt.close()


# --------------------------------------------------------
# 1.6 ACP MEDIANTE prcomp()
# --------------------------------------------------------

print("\n1.6 ACP MEDIANTE PCA (sklearn)")
print("-" * 60)

ACP_2 = PCA()
ACP_2.fit(head_dat)

print("\nPCA summary (2 variables):")
print(f"Explained variance ratio: {ACP_2.explained_variance_ratio_}")
print(f"Cumulative variance: {np.cumsum(ACP_2.explained_variance_ratio_)}")

print("\nNota: Las dos primeras componentes explican el 100% de la")
print("varianza porque la dimensión original tiene únicamente dos variables.")
print(f"La primera componente explica aproximadamente el {ACP_2.explained_variance_ratio_[0]*100:.2f}%")
print(f"mientras que la segunda explica el {ACP_2.explained_variance_ratio_[1]*100:.2f}%")


# ============================================================
# EJERCICIO 2
# ACP SOBRE LAS CUATRO VARIABLES
# ============================================================

print("\n" + "=" * 60)
print("EJERCICIO 2: ACP SOBRE LAS CUATRO VARIABLES")
print("=" * 60)

# Utilizamos todas las variables de frets
head_dat_4 = frets

# --------------------------------------------------------
# 2.1 MATRIZ DE COVARIANZAS
# --------------------------------------------------------

print("\n2.1 MATRIZ DE COVARIANZAS")
print("-" * 60)

cov_head_4 = head_dat_4.cov()
print("\nMatriz de covarianzas:")
print(cov_head_4)

# Varianzas individuales
print("\nVarianzas individuales:")
print(np.diag(cov_head_4.values))

# Varianza total
var_total_4 = np.sum(np.diag(cov_head_4.values))
print(f"\nVarianza total: {var_total_4:.6f}")


# --------------------------------------------------------
# 2.2 DESCOMPOSICIÓN ESPECTRAL
# --------------------------------------------------------

print("\n2.2 DESCOMPOSICIÓN ESPECTRAL")
print("-" * 60)

# Valores propios y vectores propios
eigenvalues_4, eigenvectors_4 = np.linalg.eig(cov_head_4.values)

# Ordenar por valores propios en orden descendente
idx_4 = eigenvalues_4.argsort()[::-1]
eigenvalues_4 = eigenvalues_4[idx_4]
eigenvectors_4 = eigenvectors_4[:, idx_4]

print("\nValores propios:")
print(eigenvalues_4)

# Comprobación de la varianza total
print(f"\nSuma de valores propios (varianza total): {np.sum(eigenvalues_4):.6f}")

# Proporción de varianza explicada por la primera CP
prop_var_pc1_4 = eigenvalues_4[0] / np.sum(eigenvalues_4)
print(f"\nProporción de varianza explicada por CP1: {prop_var_pc1_4:.6f}")


# --------------------------------------------------------
# 2.3 VECTORES PROPIOS
# --------------------------------------------------------

print("\n2.3 VECTORES PROPIOS")
print("-" * 60)

print("\nPrimer vector propio:")
print(eigenvectors_4[:, 0])

print("\nSegundo vector propio:")
print(eigenvectors_4[:, 1])


# --------------------------------------------------------
# 2.4 MEDIAS Y COORDENADAS DE LA PRIMERA OBSERVACIÓN
# --------------------------------------------------------

print("\n2.4 MEDIAS Y COORDENADAS DE LA PRIMERA OBSERVACIÓN")
print("-" * 60)

# Medias de las cuatro variables
means_head_4 = head_dat_4.mean().values
print(f"\nMedias de las cuatro variables:")
print(means_head_4)

# Primera observación
obs_1_4 = head_dat_4.iloc[0].values
print(f"\nPrimera familia:")
print(obs_1_4)

# Centramos la observación
obs_1_centered_4 = obs_1_4 - means_head_4

# Coordenada sobre la primera componente
score_pc1_4 = eigenvectors_4[:, 0] @ obs_1_centered_4
print(f"\nCoordenada sobre CP1: {score_pc1_4:.6f}")

# Coordenada sobre la segunda componente
score_pc2_4 = eigenvectors_4[:, 1] @ obs_1_centered_4
print(f"Coordenada sobre CP2: {score_pc2_4:.6f}")


# --------------------------------------------------------
# 2.5 ACP MEDIANTE PCA (sklearn)
# --------------------------------------------------------

print("\n2.5 ACP MEDIANTE PCA (sklearn)")
print("-" * 60)

ACP = PCA()
ACP.fit(head_dat_4)

print("\nPCA summary (4 variables):")
print(f"Explained variance ratio: {ACP.explained_variance_ratio_}")
print(f"Cumulative variance: {np.cumsum(ACP.explained_variance_ratio_)}")


# ============================================================
# EJERCICIO 3
# VARIANZA EXPLICADA Y REPRESENTACIÓN GRÁFICA
# ============================================================

print("\n" + "=" * 60)
print("EJERCICIO 3: VARIANZA EXPLICADA Y REPRESENTACIÓN GRÁFICA")
print("=" * 60)

# --------------------------------------------------------
# 3.1 PROPORCIÓN DE VARIANZA EXPLICADA
# --------------------------------------------------------

print("\n3.1 PROPORCIÓN DE VARIANZA EXPLICADA")
print("-" * 60)

exp_var = ACP.explained_variance_ratio_
print(f"\nProporción de varianza explicada por cada componente:")
print(exp_var)

# Gráfico de varianza explicada
fig, ax = plt.subplots(figsize=(10, 6))
components = np.arange(1, len(exp_var) + 1)
bars = ax.bar(components, exp_var, width=0.6, color='steelblue', edgecolor='black')

# Color gradiente según valor
colors = plt.cm.viridis(exp_var / exp_var.max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_ylim(0, 0.9)
ax.set_xlabel('Componente principal')
ax.set_ylabel('Proporción de varianza')
ax.set_title('Proporción de varianza explicada')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('c:\\Users\\usuario\\OneDrive\\Escritorio\\python baby\\varianza_explicada.png', dpi=150)
print("\nGráfico guardado como 'varianza_explicada.png'")
plt.close()


# --------------------------------------------------------
# 3.2 PROPORCIÓN DE VARIANZA ACUMULADA
# --------------------------------------------------------

print("\n3.2 PROPORCIÓN DE VARIANZA ACUMULADA")
print("-" * 60)

cum_var = np.cumsum(exp_var)
print(f"\nProporción de varianza acumulada:")
print(cum_var)

# Gráfico de varianza acumulada
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(components, cum_var, width=0.6, color='steelblue', edgecolor='black')

# Color gradiente según valor
colors = plt.cm.viridis(cum_var / cum_var.max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_ylim(0, 1)
ax.set_xlabel('Componente principal')
ax.set_ylabel('Proporción de varianza acumulada')
ax.set_title('Proporción acumulada de varianza explicada')
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('c:\\Users\\usuario\\OneDrive\\Escritorio\\python baby\\varianza_acumulada.png', dpi=150)
print("\nGráfico guardado como 'varianza_acumulada.png'")
plt.close()


# --------------------------------------------------------
# 3.3 VARIANZA EXPLICADA POR LAS DOS PRIMERAS CP
# --------------------------------------------------------

print("\n3.3 VARIANZA EXPLICADA POR LAS DOS PRIMERAS CP")
print("-" * 60)

print(f"\nProporción explicada por CP1: {exp_var[0]:.6f}")
print(f"Proporción explicada por CP2: {exp_var[1]:.6f}")
print(f"Proporción acumulada por las dos primeras componentes: {sum(exp_var[:2]):.6f}")

print("\nLas dos primeras componentes explican aproximadamente")
print(f"el {sum(exp_var[:2])*100:.2f}% de la varianza total.")
print("\nPor tanto, podemos reducir la dimensión de 4 variables")
print("a 2 componentes conservando aproximadamente el")
print(f"{sum(exp_var[:2])*100:.2f}% de la información.")


# --------------------------------------------------------
# 3.4 COORDENADAS DE LAS OBSERVACIONES
# --------------------------------------------------------

print("\n3.4 COORDENADAS DE LAS OBSERVACIONES")
print("-" * 60)

# Transformar los datos al espacio de componentes
scores = ACP.transform(head_dat_4)

print("\nPrimeras filas de las coordenadas:")
print(pd.DataFrame(scores, columns=[f'PC{i+1}' for i in range(scores.shape[1])]).head())

# Gráfico de las observaciones sobre las dos primeras CP
plt.figure(figsize=(10, 8))
plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, s=100, c='seagreen', edgecolors='black')
plt.xlabel('Primera componente principal')
plt.ylabel('Segunda componente principal')
plt.title('Observaciones en el espacio de las dos primeras CP')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('c:\\Users\\usuario\\OneDrive\\Escritorio\\python baby\\observaciones_cp.png', dpi=150)
print("\nGráfico guardado como 'observaciones_cp.png'")
plt.close()


# ============================================================
# EJERCICIO 4
# BIPLOT DEL ACP
# ============================================================

print("\n" + "=" * 60)
print("EJERCICIO 4: BIPLOT DEL ACP")
print("=" * 60)

# Crear biplot
fig, ax = plt.subplots(figsize=(12, 10))

# Plotear observaciones
scatter = ax.scatter(scores[:, 0], scores[:, 1], alpha=0.6, s=100, 
                     c='seagreen', edgecolors='black', linewidth=1.5)

# Plotear vectores de variables
loadings = ACP.components_.T * np.sqrt(ACP.explained_variance_)
for i, variable in enumerate(head_dat_4.columns):
    ax.arrow(0, 0, loadings[i, 0]*3, loadings[i, 1]*3,
             head_width=0.1, head_length=0.1, fc='red', ec='red', linewidth=2)
    ax.text(loadings[i, 0]*3.2, loadings[i, 1]*3.2, variable, 
            fontsize=12, ha='center', va='center', weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

ax.set_xlabel(f'PC1 ({exp_var[0]*100:.2f}%)')
ax.set_ylabel(f'PC2 ({exp_var[1]*100:.2f}%)')
ax.set_title('ACP - Biplot (Individuos y Variables)')
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linewidth=0.5)
ax.axvline(x=0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('c:\\Users\\usuario\\OneDrive\\Escritorio\\python baby\\biplot_acp.png', dpi=150)
print("\nGráfico guardado como 'biplot_acp.png'")
plt.close()


# ============================================================
# RESUMEN DE RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESUMEN DEL ACP")
print("=" * 60)

print(f"\nVarianza explicada por CP1: {exp_var[0]*100:.2f}%")
print(f"Varianza explicada por CP2: {exp_var[1]*100:.2f}%")
print(f"Varianza acumulada CP1 + CP2: {sum(exp_var[:2])*100:.2f}%")
print("\n" + "=" * 60)
