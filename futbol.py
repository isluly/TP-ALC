import numpy as np
from alc import matvec,matriz_inversa,resolverDLT2D,obtenerQ2d
import matplotlib.pyplot as plt
import os


def cancha_a_imagen(H,UV): 
    # H es la homografia que lleva de la cancha a la imagen
    # UV son los puntos (N,2) en metros:
    N = len(UV)
    res = np.zeros((N,2))
    for i in range(N):
        p = matvec(H,np.array([UV[i][0], UV[i][1], 1.0]))
        res[i][0] = p[0] / p[2]
        res[i][1] = p[1] / p[2]
    return res

def imagen_a_cancha(H,uv): #uv son los puntos(N,2) en pixeles
    Hinv = matriz_inversa(H)
    N = len(uv)
    res = np.zeros((N,2))
    for i in range(N):
        p = matvec(Hinv,np.array([uv[i][0],uv[i][1],1]))
        res[i][0] = p[0] / p[2]
        res[i][1] = p[1] / p[2]
    return res

# gol enzo, tomando el arco de la izquierda(calculados a partir de las medidas que nos dieron de la cancha)
objPts = np.array([[0,24.84],[0,43.16],[5.5,24.84],[5.5,43.16],[16.5,26.69],[16.5,41.31],[20.15,34]])
# ya calculados 
imgPts = np.array([[191,171],[843,189],[153,231],[867,251],
                   [144,405],[853,422],[506,488]], dtype=float)

Q = obtenerQ2d(imgPts,objPts)
H = resolverDLT2D(Q)

# chequeo de la proyeccion - desp se borra si quieren
rep = cancha_a_imagen(H, objPts)
print(np.abs(rep - imgPts).max())

# distancia de enzo a la linea de gol
pie_e = np.array([[808,541]]) # aca van los pixeles que me de el pie de enzo
UV = imagen_a_cancha(H,pie_e)
print("enzo en cancha :", UV, " distancia en metros a la linea de gol: ",UV[0][0])

# circulos de 1m
ruta_img = os.path.join("futbol", "test-gol-enzo-marcada.png") # reemplazar por imagen con puntos marcados en los pies y en los arcos?
img = plt.imread(ruta_img)
plt.imshow(img)
pies_arg = np.array([[808,541],[108,538], [245,398],[375,327],[852,352],[391,293]], dtype=float) # aca van los pixeles de los pies de cada jug arg
t = np.linspace(0, 2*np.pi, 100)
for UV0 in imagen_a_cancha(H,pies_arg):
    circ = np.zeros((100,2))
    for k in range(100):
        circ[k][0] = UV0[0] + np.cos(t[k])
        circ[k][1] = UV0[1] + np.sin(t[k])
    c = cancha_a_imagen(H,circ)
    plt.plot(c[:,0],c[:,1],'c')
plt.show()

# offside
# arco de la derecha
objPts2 = np.array([[88.5,13.84],[88.5,54.16],[99.5,24.84],[99.5,43.16],[105,43.16],[105,68]], dtype=float)
# algunos que me tiro claude, hay que calcularlos de la imagen y reemplazar
imgPts2 = np.array([[732, 758],[733, 315],[1364, 577],[1228, 389],[1385, 309],[1304, 235]], dtype=float)

H2 = resolverDLT2D(obtenerQ2d(imgPts2,objPts2))
rep2 = cancha_a_imagen(H2,objPts2)
# para chequear si esta bien la H2
print("error reproyección offside:", np.abs(rep2 - imgPts2).max())   # debería ser pocos px

# reemplazar por los pixeles reales de la imagen
defensores = np.array([[826,325], [816,559]], dtype=float)
UV_def = imagen_a_cancha(H2,defensores)
print("defensores en cancha:", UV_def)

# ultimo defensor, el de mayor U(mas cercano a x=105)
imax = 0
for i in range(len(UV_def)):
    if UV_def[i][0] > UV_def[imax][0]:
        imax = i
U1 = UV_def[imax][0]
print("linea de offside en U", U1,"m")

# recta paralela a la linea del fondo, U = U1,V recorre todo el ancho
V = np.linspace(0,68,50)
recta = np.zeros((50,2))
for k in range(50):
    recta[k][0] = UV_def[0][0]
    recta[k][1] = V[k]
    
r = cancha_a_imagen(H2,recta)

# atacantes, los ingleses, calcular los pixeles
atacantes = np.array([[873,592],[838,528],[862,505],[845,412]],dtype=float)
UV_at = imagen_a_cancha(H2,atacantes)
for i in range(len(UV_at)):
    if UV_at[i][0] > U1:
        estado = "offside"
    else:
        estado = "habilitado"
    print("atacante",i,"en_cancha:",UV_at[i],"estado:",estado)

ruta_img2 = os.path.join("futbol","test-fuera-de-juego-marcada.png") # reemplazar por la de los puntos marcados
img2 = plt.imread(ruta_img2)
plt.imshow(img2)
plt.plot(r[:,0],r[:,1],'c')
plt.plot(defensores[:,0],defensores[:,1],'rx')
plt.plot(atacantes[:,0],atacantes[:,1],'bx')
plt.xlim(0,img2.shape[1])
plt.ylim(img2.shape[0],0)
plt.show()
