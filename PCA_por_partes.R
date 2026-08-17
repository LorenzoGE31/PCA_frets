# ============================================================
# ANÁLISIS DE COMPONENTES PRINCIPALES - BASE DE DATOS FRETS
# ============================================================
#
# Base de datos: frets
# Paquete: boot
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


# ------------------------------------------------------------
# LIBRERÍAS Y DATOS
# ------------------------------------------------------------

library(boot)
library(ggplot2)
library(factoextra)

data("frets")

# Comprobación inicial de los datos
head(frets)
str(frets)
summary(frets)


# ============================================================
# EJERCICIO 1
# ACP SOBRE b1 Y b2
# ============================================================

# Seleccionamos las variables de anchura de la cabeza
head_dat <- frets[, c("b1", "b2")]


# ------------------------------------------------------------
# 1.1 MATRIZ DE COVARIANZAS
# ------------------------------------------------------------

cov_head <- cov(head_dat)

cov_head

# Varianzas de cada variable
diag(cov_head)

# Varianza total
var_total <- sum(diag(cov_head))

var_total


# ------------------------------------------------------------
# 1.2 DESCOMPOSICIÓN ESPECTRAL
# ------------------------------------------------------------

eig <- eigen(cov_head)

# Valores propios
eig$values

# Comprobación de la varianza total
sum(eig$values)

# Proporción de varianza explicada por la primera CP
prop_var_pc1 <- eig$values[1] / sum(eig$values)

prop_var_pc1


# ------------------------------------------------------------
# 1.3 VECTORES PROPIOS
# ------------------------------------------------------------

# Primer vector propio
eig$vectors[, 1]

# Segundo vector propio
eig$vectors[, 2]


# ------------------------------------------------------------
# 1.4 COORDENADAS DE LA PRIMERA OBSERVACIÓN
# ------------------------------------------------------------

# Medias de las variables
means_head <- colMeans(head_dat)

means_head

# Valores de la primera familia
obs_1 <- as.numeric(head_dat[1, ])

obs_1

# Centramos la primera observación
obs_1_centered <- obs_1 - means_head

# Coordenada de la primera observación sobre la CP1
score_pc1 <- eig$vectors[, 1] %*% obs_1_centered

score_pc1

# Coordenada de la primera observación sobre la CP2
score_pc2 <- eig$vectors[, 2] %*% obs_1_centered

score_pc2


# ------------------------------------------------------------
# 1.5 REPRESENTACIÓN DE LAS COMPONENTES PRINCIPALES
# ------------------------------------------------------------

# Pendiente y ordenada de la primera componente
a1 <- means_head[2] -
  eig$vectors[1, 2] * means_head[2] / eig$vectors[1, 1]

b1 <- eig$vectors[1, 2] / eig$vectors[1, 1]


# Pendiente y ordenada de la segunda componente
a2 <- means_head[2] -
  (-eig$vectors[1, 1] * means_head[2] / eig$vectors[1, 2])

b2 <- -eig$vectors[1, 1] / eig$vectors[1, 2]


# Gráfico de las observaciones y las componentes
plot(
  head_dat,
  xlab = "Anchura del primer hijo (mm)",
  ylab = "Anchura del segundo hijo (mm)",
  xlim = c(137, 167),
  ylim = c(130, 159),
  main = "ACP sobre la anchura de la cabeza"
)

# Primera componente principal
abline(a1, b1)

# Segunda componente principal
abline(a2, b2, lty = 2)


# ------------------------------------------------------------
# 1.6 ACP MEDIANTE prcomp()
# ------------------------------------------------------------

ACP_2 <- prcomp(head_dat)

summary(ACP_2)

# Las dos primeras componentes explican el 100 % de la
# varianza porque la dimensión original tiene únicamente
# dos variables.
#
# La primera componente explica aproximadamente el 85 %,
# mientras que la segunda explica el porcentaje restante.


# ============================================================
# EJERCICIO 2
# ACP SOBRE LAS CUATRO VARIABLES
# ============================================================

# Utilizamos todas las variables de frets
head_dat <- frets


# ------------------------------------------------------------
# 2.1 MATRIZ DE COVARIANZAS
# ------------------------------------------------------------

cov_head <- cov(head_dat)

cov_head

# Varianzas individuales
diag(cov_head)

# Varianza total
var_total <- sum(diag(cov_head))

var_total


# ------------------------------------------------------------
# 2.2 DESCOMPOSICIÓN ESPECTRAL
# ------------------------------------------------------------

eig <- eigen(cov_head)

# Valores propios
eig$values

# Comprobación de la varianza total
sum(eig$values)

# Proporción de varianza explicada por la primera CP
prop_var_pc1 <- eig$values[1] / sum(eig$values)

prop_var_pc1


# ------------------------------------------------------------
# 2.3 VECTORES PROPIOS
# ------------------------------------------------------------

# Primer vector propio
eig$vectors[, 1]

# Segundo vector propio
eig$vectors[, 2]


# ------------------------------------------------------------
# 2.4 MEDIAS Y COORDENADAS DE LA PRIMERA OBSERVACIÓN
# ------------------------------------------------------------

# Medias de las cuatro variables
means_head <- colMeans(head_dat)

means_head

# Primera observación
obs_1 <- as.numeric(head_dat[1, ])

obs_1

# Centramos la observación
obs_1_centered <- obs_1 - means_head

# Coordenada sobre la primera componente
score_pc1 <- eig$vectors[, 1] %*% obs_1_centered

score_pc1

# Coordenada sobre la segunda componente
score_pc2 <- eig$vectors[, 2] %*% obs_1_centered

score_pc2


# ------------------------------------------------------------
# 2.5 ACP MEDIANTE prcomp()
# ------------------------------------------------------------

ACP <- prcomp(head_dat)

summary(ACP)


# ============================================================
# EJERCICIO 3
# VARIANZA EXPLICADA Y REPRESENTACIÓN GRÁFICA
# ============================================================


# ------------------------------------------------------------
# 3.1 PROPORCIÓN DE VARIANZA EXPLICADA
# ------------------------------------------------------------

exp_var <- ACP$sdev^2 / sum(ACP$sdev^2)

exp_var


# Data frame para el gráfico
data_exp_var <- data.frame(
  componente = 1:length(exp_var),
  varianza = exp_var
)


# Gráfico de varianza explicada
ggplot(
  data_exp_var,
  aes(x = componente, y = varianza, fill = varianza)
) +
  geom_col(width = 0.5) +
  scale_y_continuous(limits = c(0, 0.9)) +
  theme_bw() +
  labs(
    title = "Proporción de varianza explicada",
    x = "Componente principal",
    y = "Proporción de varianza"
  )


# ------------------------------------------------------------
# 3.2 PROPORCIÓN DE VARIANZA ACUMULADA
# ------------------------------------------------------------

cum_var <- cumsum(exp_var)

cum_var


# Data frame para el gráfico
data_cum_var <- data.frame(
  componente = 1:length(cum_var),
  varianza_acumulada = cum_var
)


# Gráfico de varianza acumulada
ggplot(
  data_cum_var,
  aes(
    x = componente,
    y = varianza_acumulada,
    fill = varianza_acumulada
  )
) +
  geom_col(width = 0.5) +
  scale_y_continuous(limits = c(0, 1)) +
  theme_bw() +
  labs(
    title = "Proporción acumulada de varianza explicada",
    x = "Componente principal",
    y = "Proporción de varianza acumulada"
  )


# ------------------------------------------------------------
# 3.3 VARIANZA EXPLICADA POR LAS DOS PRIMERAS CP
# ------------------------------------------------------------

# Proporción explicada individualmente
exp_var[1]
exp_var[2]

# Proporción acumulada por las dos primeras componentes
sum(exp_var[1:2])


# Las dos primeras componentes explican aproximadamente
# el 90.84 % de la varianza total.
#
# Por tanto, podemos reducir la dimensión de 4 variables
# a 2 componentes conservando aproximadamente el 90.84 %
# de la información.


# ------------------------------------------------------------
# 3.4 COORDENADAS DE LAS OBSERVACIONES
# ------------------------------------------------------------

# prcomp() proporciona directamente las coordenadas de
# las observaciones en el espacio de las componentes.
scores <- ACP$x

head(scores)


# Gráfico de las observaciones sobre las dos primeras CP
plot(
  scores[, 1],
  scores[, 2],
  xlab = "Primera componente principal",
  ylab = "Segunda componente principal",
  main = "Observaciones en el espacio de las dos primeras CP",
  pch = 19
)


# ============================================================
# EJERCICIO 4
# BIPLOT DEL ACP
# ============================================================

# Biplot de individuos y variables
fviz_pca_biplot(
  ACP,
  alpha.ind = "contrib",
  col.ind = "seagreen",
  col.var = "cos2",
  gradient.cols = c(
    "#FDF50E",
    "#FD960E",
    "#FD1E0E"
  ),
  repel = TRUE
) +
  theme_bw() +
  labs(
    title = "ACP - Biplot"
  )


# ============================================================
# RESUMEN DE RESULTADOS
# ============================================================

cat("\n========================================\n")
cat("RESUMEN DEL ACP\n")
cat("========================================\n\n")

cat(
  "Varianza explicada por CP1:",
  round(exp_var[1] * 100, 2),
  "%\n"
)

cat(
  "Varianza explicada por CP2:",
  round(exp_var[2] * 100, 2),
  "%\n"
)

cat(
  "Varianza acumulada CP1 + CP2:",
  round(sum(exp_var[1:2]) * 100, 2),
  "%\n"
)

