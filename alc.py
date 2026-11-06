"""
TP 2 ÁLGEBRA LINEAL COMPUTACIONAL

GRUPO: Triangulados
INTEGRANTES: Posse Lucila, Ventroni Milena, Pili, Francisco Krause

"""

#%% IMPORTS

import numpy as np
import pandas as pd
import os #Resuelvo tema de import de archivos

#%% Funciones del MODULO ALC

def multiplicacion_matrices(A,B):
    n,k = A.shape
    k2,m = B.shape
    if k != k2:
        return None
    c = np.zeros((n,m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i,t] * B[t,j]
            c[i,j] = s
    return c

# multiplicacion matriz por vector
def matvec(A,v):
    m, n = np.shape(A)
    if (n != len(v)):
            print('No se puede')
            return None
    res = np.zeros(m)
    for i in range(m):
        res[i] = producto_escalar(A[i,:],v) # v * fila i
    return res

# multiplicacion vector por matriz
def vecmat(v,A):
    m, n = np.shape(A)
    if (len(v) != m):
        print('No se puede')
        return None

    res = np.zeros(n)
    for i in range(n):
        res[i] = producto_escalar(v,A[:,i]) # v * columna i
    return res

def es_matriz_inversible(A):
    n,m = A.shape
    if n != m:
        return False
    return True 

def matriz_inversa(A):
    if es_matriz_inversible(A):
        A = A.astype(float).copy()
        n,m = A.shape
        I = np.zeros((n,n))
        for i in range(n):
            I[i,i] = 1.0
            
        for i in range(n):
            # buscamos el pivote
            pivote = A[i,i]
            if pivote == 0:
                return None
            # normalizamos la fila del pivote
            for j in range(n):
                A[i,j] = A[i,j] / pivote
                I[i,j] = I[i,j] / pivote
            # eliminamos las otras filas
            for k in range(n):
                if k != i:
                    factor = A[k,i]
                    for j in range(n):
                        A[k,j] = A[k,j] - factor * A[i,j]
                        I[k,j] = I[k,j] - factor * I[i,j]
        return I

# creo que esta mal, hay que corregirla
def QR_con_GS(A, tol = 1e-12, retorna_nops = False):
    m,n = A.shape
    if m != n:
        return None
    
    Q = np.eye(n)
    R = np.zeros((n,n))
    nops = 0
    for i in range(n):
        v = A[:,i].copy()
        for j in range(i):
            
            R[j,i] = producto_escalar(Q[:,j],A[:,i])
            # restamos la proyeccion
            v -= R[j,i] * Q[:,j]
            nops += 2 * m + 1
        norma_v = norma(v,2)
        nops += 2 * m 
        
        if norma_v < tol:
            R[i,i] = 0
            Q[:,i] = np.zeros(m)
        else:
            R[i,i] = norma_v
            Q[:,i] = v / norma_v
            nops += m
            
    if retorna_nops:
        return Q, R, nops
    else:
        return Q, R

def calculaLDV(A):
  if A is None:
    return None

  m = len(A)
  n = len(A[0])
  if m != n:
    return None

  U = [[float(A[i][j]) for j in range(n)] for i in range(n)]
  L = [[float(i == j) for j in range(n)] for i in range(n)]

  for i in range(n):
    if abs(U[i][i]) < 1e-14:
      return None  

    for j in range(i + 1, n):
      factor = U[j][i] / U[i][i]
      L[j][i] = factor
      for k in range(i, n):
        U[j][k] = U[j][k] - factor * U[i][k]

  if abs(U[n - 1][n - 1]) < 1e-14:
    return None

  D = [[0.0] * n for _ in range(n)]
  V = [[float(i == j) for j in range(n)] for i in range(n)]

  for i in range(n):
    d_ii = U[i][i]
    if abs(d_ii) < 1e-14:
      return None
    D[i][i] = d_ii
    for j in range(i + 1, n):
      V[i][j] = U[i][j] / d_ii

  return (
      np.array(L, dtype=float),
      np.array(D, dtype=float),
      np.array(V, dtype=float),
  )

def esSDP(A, atol=1e-8):
  if A is None:
    return False

  n = len(A)
  for i in range(n):
    for j in range(n):
      if abs(A[i][j] - A[j][i]) > atol:
        return False


  ldv = calculaLDV(A)
  if ldv is None:
    return False

  _, D, _ = ldv


  for i in range(n):
    if D[i, i] <= atol:
      return False

  return True

def calculaCholesky(A, atol=1e-10):
    A = np.array(A, dtype=float)
    
    if not esSDP(A):
        return None
        
    L, D, _ = calculaLDV(A)
    
    L = np.array(L, dtype=float)
    D = np.array(D, dtype=float)
    n = len(A)

    es_2d = isinstance(D[0], (list, np.ndarray))
    
    R = np.zeros((n, n))
    for j in range(n):
        if es_2d:
            val_d = float(D[j, j])
        else:
            val_d = float(D[j])
            
        if val_d <= atol:
            return None
            
  
        raiz_dj = np.sqrt(val_d)
        
        for i in range(n):
            R[i, j] = L[i, j] * raiz_dj
            
    return R

def norma(x,p):
    if p == 'inf':
        maximo = 0
        for i in range(len(x)):
            if np.abs(x[i]) > maximo:
                maximo = np.abs(x[i])
        return maximo
    else:
        suma = 0
        for i in range(len(x)):
            suma += np.abs(x[i]) ** p
        return suma ** (1/p)

def transpuesta(A):
    resp = []
    for i in range(A.shape[1]):
        fila = []
        for j in range(A.shape[0]):
            fila.append(A[j,i])
        resp.append(fila)
    return np.array(resp)

def metpot2k(A, tol=1e-13, K=10000):
    K = int(K)
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / norma(v, 2)
    L = 0
    for k in range(K):
        w = matvec(A,v)
        indice = componente_max(w)
        L = w[indice]
        if L == 0:
            return v, 0.0, k
        v_nuevo = w / L
        if norma(v_nuevo - v, 2) < tol:
            return v_nuevo, L, k + 1
        v = v_nuevo
    return v,L,K

def componente_max(x):
    maximo = 0
    indice = -1
    for i in range(len(x)):
        if np.abs(x[i]) > maximo:
            maximo = np.abs(x[i])
            indice = i
    return indice

def producto_interno(x, y):
    n = len(x)
    m = len(y)
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = x[i] * y[j]
    return res

def producto_escalar(v,w):
    if (len(v) != len(w)):
        print('No se puede')
        return None

    res = 0
    for i in range(0, len(v)):
        res += v[i]*w[i]

    return res

def diagRH(A, tol=1e-15, K=1e5):
    n = A.shape[0]
    v1, l1, _ = metpot2k(A, tol, K)
    v1 = v1 / norma(v1, 2)            

    e1 = np.zeros(n)
    e1[0] = 1
    u = e1 - v1
    nu = norma(u, 2)
    if nu < 1e-14:                     
        H = np.eye(n)
    else:
        H = np.eye(n) - 2 * producto_interno(u,u) / nu**2

    if n == 2:
        S = H
        D = multiplicacion_matrices(multiplicacion_matrices(H, A), H.T)
    else:
        B = multiplicacion_matrices(multiplicacion_matrices(H, A), H.T)
        A_tilde = B[1:, 1:]
        S_tilde, D_tilde = diagRH(A_tilde, tol, K)

        D = np.zeros((n, n))
        D[0, 0] = l1
        D[1:, 1:] = D_tilde

        S_blk = np.zeros((n, n))
        S_blk[0, 0] = 1
        S_blk[1:, 1:] = S_tilde
        S = multiplicacion_matrices(H, S_blk)
    return S, D

def res_tri(L,b, inferior=True, tol=1e-12):
    #asumamos que es inferior. tiene esta pinta:
    # [* 0 0 0]
    # [* * 0 0]
    # [* * * 0]
    # [* * * *] ...etc para nxn
    n, m = np.shape(L)
    if n!=m:
        print('Matriz no cuadrada')
        return None
    if len(b) != n:
       print('no se puede')
       return None

    x = np.zeros(n)
    inicio = -1
    fin = -1
    paso = 0

    # for (int i = inicio, i {hasta} fin, i += paso)
    if inferior:
        inicio = 0
        fin = n
        paso = 1
    else:
        inicio = n-1
        fin = -1
        paso = -1
    # x0 es facilito
    if abs(L[inicio][inicio]) > tol: # no hay nada que me diga que esto no es (casi) cero...
        x[inicio] = b[inicio] / L[inicio][inicio]
    elif abs(b[inicio]) < tol:
        x[inicio] = 0 # en realidad tendria variable libre pero bueno
    else: # Abs!
        print('0=' + str(b[inicio]) + ' - Abs! Sistema incompatible')
        return None

    # ahora a partir de eso hago los proximos
    for i in range(inicio+paso,fin,paso):
        # calculo lo que ya tengo resuelto
        bPrima = 0
        for j in range(inicio,i,paso):
            bPrima += L[i][j]*x[j]

        # resuelvo la incognita nueva
        bPrima = b[i] - bPrima
        if abs(L[i][i]) > tol: # no hay nada que me diga que esto no es (casi) cero...
            x[i] = bPrima / L[i][i]
        elif abs(bPrima) < tol:
            x[i] = 0
        else: # Abs!
            print('0=' + str(bPrima) + ' - Abs! Sistema incompatible')
            return None
    return x



# aplicaciones en notebook
# ejercicio 2
def obtenerQ3d(imgPts, objPts):
    N = len(objPts)
    Q = np.zeros((2 * N, 12))
    for i in range(N):
        u, v = imgPts[i][0], imgPts[i][1]
        P = [objPts[i][0], objPts[i][1], objPts[i][2], 1.0]
        # por cada coordenada del punto 3D, se agregan dos filas a Q
        for k in range(4):
            # va armando Q por grupos de columnas, cada grupo de columnas corresponden a un punto 3D
            # columnas 0-3 : P[k]
            Q[2*i,k] = P[k]
            # columnas 4-7 : 0
            Q[2*i, 4 + k] = 0.0
            # columnas 8-11: -u*P[k]
            Q[2*i, 8 + k] = -u * P[k]
            
            # lo mismo para la segunda fila
            Q[2*i + 1, k] = 0.0
            Q[2*i + 1, 4 + k] = P[k]
            Q[2*i + 1, 8 + k] = -v * P[k]
    return Q


def resolverDLT3D(Q):
    A = multiplicacion_matrices(transpuesta(Q), Q)
    S,D = diagRH(A)
    m = S[:, -1] # ultima columna de S, que corresponde al vector singular del menor valor singular
    m = m / norma(m,2) # normaliza el vector? 
    # extender m a 3x4
    return m.reshape(3,4)

def espejar(A, filas):
    if (filas):
        return A[::-1] # equivalente a hacer PA con P la matriz identidad espejada
    else:
        return A[:,::-1] # equivalente a hacer AP

def devolverMatricesCalibracion3d(M, restric=None): # restric puede ser None, 'skew', 'alfa', 'skew-alfa' (para pedir las dos cosas)
    # quiero M = K [R t] donde K es triangular superior, R es ortogonal

    if (np.shape(M) != (3,4)):
        print('M no cumple el tamaño de la matriz de proyeccion pedida')
        return None, None, None

    if (M[2][3] < 0): # corrijo signo
        M = -M

    Mprima = M[:,:3]
    MprimaT = transpuesta(Mprima)
    P = np.array([[0,0,1],[0,1,0],[1,0,0]])

    Q, R = QR_con_GS(multiplicacion_matrices(MprimaT,P)) # QR sobre M't P

    # K = P Rt P (espejando me ahorro las multiplicaciones de matrices, total ya se como permuta multiplicar por P)
    Kres = espejar(espejar(transpuesta(R), True), False).copy()
    # R = P Qt
    Rres = espejar(transpuesta(Q),True).copy()

    if restric is not None:
        if restric in ('skew', 'skew-alfa'): # pido skew = 0
            Kres[0][1] = 0.0
        if restric in ('alfa', 'skew-alfa'): # pido alfau=alfav (saco un promedio entre los dos para no tener que elegir uno u otro)
            Kres[0][0] = (Kres[0][0] + Kres[1][1]) / 2
            Kres[1][1] = Kres[0][0]

        # calculo R con este nuevo K para garantizar K R = Mprima
        Rres = np.zeros((3, 3))
        for j in range(3):
            Rres[:, j] = res_tri(Kres, Mprima[:, j], inferior=False)

        # ortonormalizo R para cumplir enunciado. notar que con esto K R aproxima Mprima, no es exacto
        r3 = Rres[2] / norma(Rres[2], 2)
        r2 = Rres[1] - producto_escalar(Rres[1], r3)*r3
        r2 = r2 / norma(r2, 2)
        r1 = Rres[0] - producto_escalar(Rres[0], r3)*r3 - producto_escalar(Rres[0], r2)*r2
        r1 = r1 / norma(r1, 2)
        Rres = np.array([r1, r2, r3])


    # resuelvo K t = {columna 4 de M} para conseguir t
    t = res_tri(Kres, M[:,3], inferior=False)
    
    # normalizo Kres para que Kres[2][2] = 1
    Kres = Kres / Kres[2][2]

    return Rres, t, Kres


# aplicacion a datos que nos dieron, ej 2

# puntos del mundo real de la forma(U,V,W) en cm?
# matriz de 12 x 3
objPts = np.array([
    [0, 8, 0],
    [10.4, 8, 0],
    [0, 21.5, 0],
    [10.4, 21.5, 0],
    [0, 26.6, -10.95],
    [10.4, 26.6, -10.95],
    [0, 26.6, -24.45],
    [10.4, 26.6, -24.45],
    [19.4, 14.6, -10.95],
    [19.4, 14.6, -24.45],
    [19.4, 25, -10.95],
    [19.4, 25, -24.45]
])
# pixeles [0,8,0] corresponde a los primeros valores que son u1v1
# leemos el archivo, nos devuelve una tabla de 24 números por fila, cada fila es un cuadro(la primera corresponde a los 12 puntos de obj_pts)
#Esto debería funciona en todas las compus

ruta_datos = os.path.join("calib", "3d", "detected_corners.txt")
datos = np.loadtxt(ruta_datos) # aca cada fila es una imagen con 24 columnas(asumo que estan u1v1u2v2..)


def imgpts_del_cuadro(k):
    fila = datos[k]
    imgPts = np.zeros((12,2))
    for i in range(12):
        imgPts[i,0] = fila[2*i] #ui
        imgPts[i,1] = fila[2*i + 1] #vi
    return imgPts

imgPts = imgpts_del_cuadro(0)
Q = obtenerQ3d(imgPts,objPts)
M = resolverDLT3D(Q)
R,T,K = devolverMatricesCalibracion3d(M)



# ejercicio 3

def obtenerQ2d(imgPts, objPts):
    N = len(objPts)
    Q = np.zeros((2 * N, 9))
    for i in range(N):
        u, v = imgPts[i][0], imgPts[i][1]
        P = [objPts[i][0], objPts[i][1], 1.0]
        # por cada coordenada del punto 2D, se agregan dos filas a Q
        for k in range(3):
            # va armando Q por grupos de columnas, cada grupo de columnas corresponden a un punto 2D
            # columnas 0-2 : P[k]
            Q[2*i,k] = P[k]
            # columnas 3-5 : 0
            Q[2*i, 3 + k] = 0.0
            # columnas 6-8: -u*P[k]
            Q[2*i, 6 + k] = -u * P[k]
            
            # lo mismo para la segunda fila
            Q[2*i + 1, k] = 0.0
            Q[2*i + 1, 3 + k] = P[k]
            Q[2*i + 1, 6 + k] = -v * P[k]
    return Q

def v(H,i,j):
    # i , j son los indices de las columnas de H
    hi, hj = H[:,i], H[:,j]
    return np.array([
        hi[0]*hj[0], # esto seria ci1 * cj1
        hi[0]*hj[1] + hi[1]*hj[0], # ci1*cj2 + ci2*cj1
        hi[1]*hj[1],
        hi[0]*hj[2] + hi[2]*hj[0],
        hi[1]*hj[2] + hi[2]*hj[1],
        hi[2]*hj[2],
    ])
    
def armarA(Hlist):
    N = len(Hlist)
    A = np.zeros((2*N,6))
    for i in range(N):
        H = Hlist[i]
        A[2*i] = v(H,0,1)
        A[2*i + 1] = v(H,0,0) - v(H,1,1)
    return A

def calcularMatrizK2D(A):
    AtA = multiplicacion_matrices(transpuesta(A),A) 
    S,D = diagRH(AtA) # busco b chico tq Ab = 0 y propongo A = SDV
    b = S[:,-1] # lo mismo que el anterior, minimizando b es la ultima columna de s
    b = b / norma(b,2) # normalizo a norma 1
    
    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if B[0,0] < 0:
        B = -B # quiero que sea positiva para usar cholesky
        
    L = calculaCholesky(B) # B = L L.T y si B = K.T(-1) K(-1) entonces K = L.T a la (-1)
    if L is None:
        return None
    K = matriz_inversa(transpuesta(L))
    return K / K[2,2] # se que K[2,2] real tiene que dar 1 por la forma de k, como cholesky puede dar una escala de K, lo divido por ese numero de la esquina asi queda en 1

# devuelve matriz h
def resolverDLT2D(Q):
    A = multiplicacion_matrices(transpuesta(Q), Q)
    S,D = diagRH(A)
    h = S[:, -1] # ultima columna de S, que corresponde al vector singular del menor valor singular
    h = h / norma(h,2) # normaliza el vector
    # extender m a 3x4
    return h.reshape(3,3)


def producto_cruz(a, b):
    return np.array([a[1]*b[2] - a[2]*b[1],a[2]*b[0] - a[0]*b[2],a[0]*b[1] - a[1]*b[0]])


def obtenerPose2D(Hj,K):
    K_inv = matriz_inversa(K)
    c1,c2,c3 = Hj[:,0], Hj[:,1], Hj[:,2]
    # viene de la formula landa(u,v,1) = K[r1,r2,t][U,V,1].T = H[U,V,1].T
    landa = 1/ norma(matvec(K_inv,c1), 2)
    r1 = landa * (matvec(K_inv,c1))
    r2 = landa * (matvec(K_inv,c2))
    t = landa *(matvec(K_inv,c3))
    
    # el tablero debe quedar adelante de la camara entonces tz > 0
    if t[2] < 0:
        r1,r2,t = -r1,-r2,-t
        
    r3 = producto_cruz(r1,r2) # garantiza ortogonalidad
    
    # construimos R
    R = np.zeros((3,3))
    R[:,0],R[:,1],R[:,2] = r1,r2,r3
    
    # puede no quedar ortogonal por el ruido, ortononormalizamos con GS
    resultado = QR_con_GS(R)
    if resultado is not None:
        R,_ = resultado # R seria la Q
        
    return R,t  

def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interes (de m x n)
    k el numero de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna U_red (matriz de m x k), valores_sing (vector de k valores singulares) y V_red (matriz de n x k)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # si m < n trabajamos con A^t
    traspuesta = m < n
    if traspuesta:
        A = transpuesta(A)
        m, n = n, m

    # autovectores de A^t A = columnas de V, autovalores = valores singulares al cuadrado
    AtA = multiplicacion_matrices(transpuesta(A), A)
    if n == 1:
        S, D = np.ones((1, 1)), AtA          # diagRH no contempla el caso 1x1
    else:
        S, D = diagRH(AtA)
    autovalores = [D[i, i] for i in range(n)]

    # diagRH devuelve primero los autovalores no nulos, de mayor a menor
    # Los asserts fallan si no se cumple
    r = len([a for a in autovalores if a > tol])
    assert all(a > tol for a in autovalores[:r]), "diagRH no devolvio primero los autovalores no nulos"
    assert all(autovalores[i] >= autovalores[i + 1] * (1 - 1e-10) for i in range(r - 1)), \
        "diagRH no devolvio los autovalores ordenados"

    # nos quedamos con los primeros k autovalores no nulos 
    cant = r if k == "max" else min(k, r)

    valores_sing = np.zeros(cant)
    V_red = np.zeros((n, cant))
    for j in range(cant):
        valores_sing[j] = np.sqrt(autovalores[j])
        V_red[:, j] = S[:, j]

    # A V = U Sigma: U_red se obtiene normalizando las columnas de A V_red
    AV = multiplicacion_matrices(A, V_red)
    U_red = np.zeros((m, cant))
    for j in range(cant):
        U_red[:, j] = AV[:, j] / norma(AV[:, j], 2)

    if traspuesta:
        return V_red, valores_sing, U_red
    return U_red, valores_sing, V_red

"""
# aplicacion al ejercicio
# objPts del tablero: esquinas(30*i, 30*j), i=0..10, j=0..6
objPts = []
for j in range(7): # cant filas(j va de 0 a 6) con borde interno
    for i in range(11): # lo mismo pero columnas
        objPts.append([30.0*i,30.0*j,0.0]) # nos armamos el punto segun enunciado
        
objPts = np.array(objPts)

#nos armamos nuestro df con esquinas marcadas
ruta_esquinas = os.path.join("calib", "coplanar", "marcadas", "esquinas.csv")
imagenes = pd.read_csv(ruta_esquinas)

Hlist = []
for nombre in imagenes["imagen"].unique(): #no usé esto porque creo que no está permitido unique... si está modificar el notebook
    d = imagenes[imagenes["imagen"] == nombre].sort_values('punto')
    imgPts = d[["u","v"]].to_numpy() 
    
    Q = obtenerQ2d(imgPts,objPts)
    H = resolverDLT2D(Q)
    Hlist.append(H)
    
# armamos A con todas las homografias
A = armarA(Hlist)

# calculamos la matriz de parametros intrinsecos K
K = calcularMatrizK2D(A)

# calculamos la pose(R,t) para cada imagen del tablero
poses = []
for i,H in enumerate(Hlist):
    R,t = obtenerPose2D(H,K)
    poses.append((R,t))
    """
