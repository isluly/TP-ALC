# Ejercicio 5.a - Datos de calibración y homografías de cada vista
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import alc

np.set_printoptions(precision=2, suppress=True)

# 1. Cargamos las vistas de calibración y los puntos del v5
vistas = pd.read_csv("salguero/world_img_pts.csv", sep=";")
cantVistas = 4
cantPuntos = len(vistas)

datosV5 = np.loadtxt("salguero/gs_10_09_point_correspondences_v5.txt")
mundoV5 = alc.transpuesta(datosV5[1:3, :])
pixelesV5 = alc.transpuesta(datosV5[4:6, :])

# 2. Tomamos las coordenadas del punto del v5 con el píxel más cercano en la vista 0
mundoCsv = np.asarray(vistas[["X", "Y"]], dtype=float)
pixelesVista0 = np.asarray(vistas[["u_0", "v_0"]], dtype=float)
mundo = np.zeros((cantPuntos, 2))

for i in range(cantPuntos):
    masCercano = 0
    for k in range(len(pixelesV5)):
        if alc.norma(pixelesV5[k] - pixelesVista0[i], 2) < alc.norma(pixelesV5[masCercano] - pixelesVista0[i], 2):
            masCercano = k
    mundo[i] = mundoV5[masCercano]
    distancia = alc.norma(pixelesV5[masCercano] - pixelesVista0[i], 2)
    print("Punto", i, "- CSV:", mundoCsv[i], "-> v5:", mundo[i], "(a %.1f px)" % distancia)

# 3. Estimamos la homografía de cada vista
Hs = []
for j in range(cantVistas):
    pixelesVista = np.asarray(vistas[["u_" + str(j), "v_" + str(j)]], dtype=float)
    Hs.append(alc.resolverDLT2D(alc.obtenerQ2d(pixelesVista, mundo)))

# Ejercicio 5.a - Cálculo de K

# 1. Zhang sin restricciones, como en el ejercicio 3
print("K con Zhang sin restricciones:")
print(alc.calcularMatrizK2D(alc.armarA(Hs)))

# 2. Punto principal en el centro de la imagen
imagenInicial = mpimg.imread("salguero/img_10_18_47_351.jpg")
u0 = imagenInicial.shape[1] / 2
v0 = imagenInicial.shape[0] / 2
centrarPixeles = np.array([[1, 0, -u0], [0, 1, -v0], [0, 0, 1]], dtype=float)

# 3. Dos ecuaciones p + f^2 q = 0 por vista
p = np.zeros(2 * cantVistas)
q = np.zeros(2 * cantVistas)
for j in range(cantVistas):
    Hcentrada = alc.multiplicacion_matrices(centrarPixeles, Hs[j])
    Hcentrada = Hcentrada / alc.norma(Hcentrada.reshape(9), 2)
    c1 = Hcentrada[:, 0]
    c2 = Hcentrada[:, 1]
    p[2*j] = c1[0]*c2[0] + c1[1]*c2[1]
    q[2*j] = c1[2]*c2[2]
    p[2*j + 1] = c1[0]**2 + c1[1]**2 - c2[0]**2 - c2[1]**2
    q[2*j + 1] = c1[2]**2 - c2[2]**2

# 4. Cuadrados mínimos en la incógnita 1/f^2
unoSobreFCuadrado = -np.sum(p * q) / np.sum(p * p)
f = 1 / np.sqrt(unoSobreFCuadrado)

K = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]], dtype=float)
print("\nK con skew = 0, alfa_u = alfa_v y punto principal en el centro:")
print(K)

# Ejercicio 5.b - Pose de la cámara y reproyección sobre img_10_18_47_351.jpg

# 1. Homografía de la calle con los 51 puntos
H0 = alc.resolverDLT2D(alc.obtenerQ2d(pixelesV5, mundoV5))

# 2. Pose con la K del inciso a)
R0, t0 = alc.obtenerPose2D(H0, K)
print("Matriz R:")
print(R0)
print("\nVector t (m):")
print(t0)
print("\nR^T R:")
print(alc.multiplicacion_matrices(alc.transpuesta(R0), R0))

# 3. Matriz de proyección P0 = K [R | t]
Rt0 = np.zeros((3, 4))
Rt0[:, 0:3] = R0
Rt0[:, 3] = t0
P0 = alc.multiplicacion_matrices(K, Rt0)

# 4. Reproyección de los 51 puntos con P0 y con H0
cantPuntosV5 = len(mundoV5)
reproyectados = np.zeros((cantPuntosV5, 2))
errorConP0 = np.zeros(cantPuntosV5)
errorConH0 = np.zeros(cantPuntosV5)
for i in range(cantPuntosV5):
    proyeccionP0 = alc.matvec(P0, np.array([mundoV5[i, 0], mundoV5[i, 1], 0, 1]))
    proyeccionH0 = alc.matvec(H0, np.array([mundoV5[i, 0], mundoV5[i, 1], 1]))
    reproyectados[i] = proyeccionP0[:2] / proyeccionP0[2]
    errorConP0[i] = alc.norma(reproyectados[i] - pixelesV5[i], 2)
    errorConH0[i] = alc.norma(proyeccionH0[:2] / proyeccionH0[2] - pixelesV5[i], 2)

print("\nError de reproyección con K [R | t]: medio = %.1f px, máximo = %.1f px" % (np.mean(errorConP0), np.max(errorConP0)))
print("Error de reproyección con H0:        medio = %.1f px, máximo = %.1f px" % (np.mean(errorConH0), np.max(errorConH0)))

# 5. Error medio por zona de la calle
Y = mundoV5[:, 1]
sendaSalguero = (Y >= 0) * (Y <= 3)
sendaGuemes = Y < 0
puntosLejanos = Y > 3
print("\nError medio por zona, con K [R | t] y con H0:")
print("Senda de Salguero: %.1f px y %.1f px" % (np.mean(errorConP0[sendaSalguero]), np.mean(errorConH0[sendaSalguero])))
print("Senda de Güemes:   %.1f px y %.1f px" % (np.mean(errorConP0[sendaGuemes]), np.mean(errorConH0[sendaGuemes])))
print("Puntos lejanos:    %.1f px y %.1f px" % (np.mean(errorConP0[puntosLejanos]), np.mean(errorConH0[puntosLejanos])))

# 6. Gráfico de la reproyección
plt.figure(figsize=(12, 9))
plt.imshow(imagenInicial)
for i in range(cantPuntosV5):
    plt.plot([pixelesV5[i, 0], reproyectados[i, 0]], [pixelesV5[i, 1], reproyectados[i, 1]], color="yellow")
plt.scatter(pixelesV5[:, 0], pixelesV5[:, 1], s=40, facecolors="none", edgecolors="lime", label="puntos originales")
plt.scatter(reproyectados[:, 0], reproyectados[:, 1], s=12, color="red", label="reproyectados con K [R | t]")
plt.legend()
plt.title("Ejercicio 5.b: reproyección sobre img_10_18_47_351.jpg")
plt.show()
def imagenACalle(Hinversa, pixel):
    '''
    Lleva un píxel al plano de la calle.
    Hinversa: inversa de la homografía de la calle a la imagen. pixel: array (u, v).
    Devuelve la posición (X, Y) en metros.
    '''
    puntoHomogeneo = alc.matvec(Hinversa, np.array([pixel[0], pixel[1], 1]))
    return puntoHomogeneo[:2] / puntoHomogeneo[2]

def pendienteCuadradosMinimos(t, x):
    '''
    Devuelve la pendiente de la recta que ajusta x(t) por cuadrados mínimos.
    t, x: arrays de igual longitud.
    '''
    return np.sum((t - np.mean(t)) * (x - np.mean(x))) / np.sum((t - np.mean(t)) ** 2)

# H = K [r1 r2 t]
Hcalle = np.zeros((3, 3))
Hcalle[:, 0] = P0[:, 0]
Hcalle[:, 1] = P0[:, 1]
Hcalle[:, 2] = P0[:, 3]
HcalleInversa = alc.matriz_inversa(Hcalle)
H0inversa = alc.matriz_inversa(H0)

# Píxeles marcados a mano de cada agente
marcados = pd.read_csv("salguero/marcados.csv")
agentes = ["taxi", "camioneta", "senora"]
tiempos = {}
posiciones = {}
velocidadesCuadro = {}
velocidadesCM = {}

for agente in agentes:
    marcasAgente = marcados[marcados["agente"] == agente]
    t = np.asarray(marcasAgente["t"], dtype=float)
    t = t - t[0]
    pixeles = np.asarray(marcasAgente[["u", "v"]], dtype=float)

    # Posiciones sobre la calle
    P = np.zeros((len(t), 2))
    P_H0 = np.zeros((len(t), 2))
    for k in range(len(t)):
        P[k] = imagenACalle(HcalleInversa, pixeles[k])
        P_H0[k] = imagenACalle(H0inversa, pixeles[k])

    # Velocidad cuadro a cuadro, mediante cuadrados mínimos
    desplazamientos = np.diff(P, axis=0)
    intervalos = np.diff(t)
    vCuadro = np.zeros(len(intervalos))
    for k in range(len(intervalos)):
        vCuadro[k] = alc.norma(desplazamientos[k], 2) / intervalos[k]

    vCM = np.sqrt(pendienteCuadradosMinimos(t, P[:, 0])**2 + pendienteCuadradosMinimos(t, P[:, 1])**2)
    vCM_H0 = np.sqrt(pendienteCuadradosMinimos(t, P_H0[:, 0])**2 + pendienteCuadradosMinimos(t, P_H0[:, 1])**2)

    tiempos[agente] = t
    posiciones[agente] = P
    velocidadesCuadro[agente] = vCuadro
    velocidadesCM[agente] = vCM

    print(agente + ":")
    print("  velocidad cuadro a cuadro (km/h):", vCuadro * 3.6)
    print("  velocidad por cuadrados mínimos: %.1f km/h (control con H0: %.1f km/h)" % (vCM * 3.6, vCM_H0 * 3.6))

# Ejercicio 5.c - Trayectorias y velocidad en el tiempo
colores = {"taxi": "gold", "camioneta": "deepskyblue", "senora": "red"}

# 1. Trayectorias en vista de planta
plt.figure(figsize=(6, 8))
plt.scatter(mundoV5[:, 0], mundoV5[:, 1], s=10, color="gray", label="puntos del v5")
for agente in agentes:
    plt.plot(posiciones[agente][:, 0], posiciones[agente][:, 1], "o-", color=colores[agente], label=agente)
plt.legend()
plt.title("Vista de planta (X, Y en metros)")
plt.show()

# 2. Velocidad en función del tiempo
fig, axs = plt.subplots(1, 3, figsize=(16, 4))
for i in range(len(agentes)):
    agente = agentes[i]
    t = tiempos[agente]
    instantesMedios = (t[:-1] + t[1:]) / 2
    vCM = velocidadesCM[agente] * 3.6
    axs[i].plot(instantesMedios, velocidadesCuadro[agente] * 3.6, "o-", color=colores[agente], label="cuadro a cuadro")
    axs[i].plot([0, t[-1]], [vCM, vCM], "k--", label="cuadrados mínimos: %.1f km/h" % vCM)
    axs[i].set_title(agente + ": velocidad (km/h) vs tiempo (s)")
    axs[i].legend()
plt.tight_layout()
plt.show()
