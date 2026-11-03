import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import alc

# ejercicio 5a) - conseguir los parametros intrinsecos a partir de world_img_pts.csv
print('----- EJERCICIO 5a -----')

ptsMuestra = pd.read_csv("salguero/world_img_pts.csv", sep=";")
muestraPtsMundo = alc.transpuesta(np.array([ptsMuestra["X"], ptsMuestra["Y"], ptsMuestra["Z"]]))
print('puntos del mundo real:')
print(muestraPtsMundo)

Hs = np.array([np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))])
for j in range(0,4):
    muestraPtsImg = alc.transpuesta(np.array([ptsMuestra['u_'+str(j)], ptsMuestra['v_'+str(j)]]))
    Q = alc.obtenerQ2d(muestraPtsImg,muestraPtsMundo)
    Hs[j] = alc.resolverDLT2D(Q)
    print('homografia j=' + str(j) + ':')
    print(Hs[j])

A = alc.armarA(Hs)
K = alc.calcularMatrizK2D(A) # consegui K!!
print('parametros intrinsecos:')
print(K)

# ejercicio 5b) - conseguir la pose de la _351.jpg (la vista j=0) y reproyectar
print('----- EJERCICIO 5b -----')

R0, t0 = alc.obtenerPose2D(Hs[0], K)
print('matriz de rotacion R para _351.jpg:')
print(R0)
print('vector de traslacion t para _351.jpg:')
print(t0)

Rt = R0.copy()
Rt[:,2] = t0
KRt = alc.multiplicacion_matrices(K,Rt)

ptsVistaCero = np.loadtxt("salguero/gs_10_09_point_correspondences_v5.txt")
reproyeccion = np.zeros((2,51))

for i in range(0,51):
    p = np.array([ptsVistaCero[1, i], ptsVistaCero[2, i], 1.0])   # (X, Y, 1)
    proy = alc.matvec(KRt, p)
    proy = proy / proy[2]
    reproyeccion[:,i] = proy[:2]
    print('punto i=' + str(i) + ': ' + str(p))
    dist = np.sqrt(abs(proy[0] - ptsVistaCero[4, i])**2 + abs(proy[1] - ptsVistaCero[5, i])**2)
    print('proyecto a '+ str(reproyeccion[:,i]) + ' --- og: ' + str(ptsVistaCero[4:, i]) + ' --- dist: ' + str(dist))

# PENDIENTE debuggear y/o analizar por que da tan mal K

# ejercicio 5c) - analizar drive de 46 imagenes donde mis puntos son la señora y el taxi en cada img

# PENDIENTE conseguir los pts de las img. tanto taxi como senora estan en Z=0 asi que todo bien
# senora2D = np.array([[3,4,1], [3,4,1], [3,4,1]]) # [u,v,1]
# taxi2D = np.array([[3,4,1], [3,4,1], [3,4,1]])

# cantImg = 46
# senora3D = np.zeros((cantImg,4))
# taxi3D = np.zeros((cantImg,4))

# KRt_inv = alc.matriz_inversa(KRt)
# for i in range(0,cantImg):
#     senora2D[i] = alc.vecmat(senora3D[i],KRt_inv)
#     taxi2D[i] = alc.vecmat(taxi3D[i],KRt_inv)

# PENDIENTE graficar y analizar para el note