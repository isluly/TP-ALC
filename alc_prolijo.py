"""
TP 2 - ÁLGEBRA LINEAL COMPUTACIONAL

Grupo: Triangulados
Integrantes: Posse Lucila, Ventroni Milena, Pili, Fran
"""

#%% Imports

import numpy as np
import pandas as pd
import os  # para armar rutas de archivos que funcionen en cualquier compu


#%% Módulo ALC - Operaciones básicas

def producto_escalar(v, w):
    """Producto escalar <v, w>."""
    if (len(v) != len(w)):
        print('No se puede')
        return None

    res = 0
    for i in range(0, len(v)):
        res += v[i] * w[i]

    return res


def producto_interno(x, y):
    """Producto externo x y^T (matriz de len(x) x len(y))."""
    n = len(x)
    m = len(y)
    res = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            res[i, j] = x[i] * y[j]
    return res


def producto_cruz(a, b):
    """Producto vectorial a x b (vectores de R^3)."""
    return np.array([a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]])


def norma(x, p):
    """Norma p de un vector. Con p = 'inf' calcula la norma infinito."""
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


def componente_max(x):
    """Índice de la componente de mayor módulo de x (-1 si x es nulo)."""
    maximo = 0
    indice = -1
    for i in range(len(x)):
        if np.abs(x[i]) > maximo:
            maximo = np.abs(x[i])
            indice = i
    return indice


def transpuesta(A):
    """Transpuesta de A."""
    resp = []
    for i in range(A.shape[1]):
        fila = []
        for j in range(A.shape[0]):
            fila.append(A[j, i])
        resp.append(fila)
    return np.array(resp)


def matvec(A, v):
    """Producto matriz por vector: A v."""
    m, n = np.shape(A)
    if (n != len(v)):
        print('No se puede')
        return None
    res = np.zeros(m)
    for i in range(m):
        res[i] = producto_escalar(A[i, :], v)  # fila i * v
    return res


def vecmat(v, A):
    """Producto vector por matriz: v^T A."""
    m, n = np.shape(A)
    if (len(v) != m):
        print('No se puede')
        return None

    res = np.zeros(n)
    for i in range(n):
        res[i] = producto_escalar(v, A[:, i])  # v * columna i
    return res


def multiplicacion_matrices(A, B):
    """Producto A B. Devuelve None si las dimensiones no son compatibles."""
    n, k = A.shape
    k2, m = B.shape
    if k != k2:
        return None
    c = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            s = 0.0
            for t in range(k):
                s += A[i, t] * B[t, j]
            c[i, j] = s
    return c


#%% Módulo ALC - Inversa y sistemas triangulares

def es_matriz_inversible(A):
    """Por ahora solo chequea que A sea cuadrada."""
    n, m = A.shape
    if n != m:
        return False
    return True


def matriz_inversa(A):
    """
    Inversa por Gauss-Jordan (sin pivoteo).
    Devuelve None si A no es cuadrada o si aparece un pivote nulo.
    """
    if es_matriz_inversible(A):
        A = A.astype(float).copy()
        n, m = A.shape
        I = np.zeros((n, n))
        for i in range(n):
            I[i, i] = 1.0

        for i in range(n):
            # pivote
            pivote = A[i, i]
            if pivote == 0:
                return None
            # normalizamos la fila del pivote
            for j in range(n):
                A[i, j] = A[i, j] / pivote
                I[i, j] = I[i, j] / pivote
            # anulamos la columna i en el resto de las filas
            for k in range(n):
                if k != i:
                    factor = A[k, i]
                    for j in range(n):
                        A[k, j] = A[k, j] - factor * A[i, j]
                        I[k, j] = I[k, j] - factor * I[i, j]
        return I


def res_tri(L, b, inferior=True, tol=1e-12):
    """
    Resuelve L x = b con L triangular (inferior por defecto, superior si inferior=False).

    Triangular inferior (n x n):
        [* 0 0 0]
        [* * 0 0]
        [* * * 0]
        [* * * *]
    """
    n, m = np.shape(L)
    if n != m:
        print('Matriz no cuadrada')
        return None
    if len(b) != n:
        print('no se puede')
        return None

    x = np.zeros(n)
    inicio = -1
    fin = -1
    paso = 0

    # equivalente a: for (int i = inicio; i != fin; i += paso)
    if inferior:
        inicio = 0
        fin = n
        paso = 1
    else:
        inicio = n-1
        fin = -1
        paso = -1

    # la primera incógnita sale directo
    if abs(L[inicio][inicio]) > tol:  # el pivote no es (casi) cero
        x[inicio] = b[inicio] / L[inicio][inicio]
    elif abs(b[inicio]) < tol:
        x[inicio] = 0  # en realidad sería una variable libre
    else:  # 0 = b != 0 -> absurdo
        print('0=' + str(b[inicio]) + ' - Abs! Sistema incompatible')
        return None

    # el resto, usando las incógnitas ya resueltas
    for i in range(inicio+paso, fin, paso):
        # parte ya conocida de la ecuación i
        bPrima = 0
        for j in range(inicio, i, paso):
            bPrima += L[i][j] * x[j]

        # despejamos la incógnita nueva
        bPrima = b[i] - bPrima
        if abs(L[i][i]) > tol:  # el pivote no es (casi) cero
            x[i] = bPrima / L[i][i]
        elif abs(bPrima) < tol:
            x[i] = 0
        else:  # 0 = bPrima != 0 -> absurdo
            print('0=' + str(bPrima) + ' - Abs! Sistema incompatible')
            return None
    return x


#%% Módulo ALC - Factorizaciones

def QR_con_GS(A, tol=1e-12, retorna_nops=False):
    """
    Factorización QR de A (cuadrada) con Gram-Schmidt clásico.
    Si retorna_nops=True devuelve además la cantidad de operaciones.
    """
    m, n = A.shape
    if m != n:
        return None

    Q = np.eye(n)
    R = np.zeros((n, n))
    nops = 0
    for i in range(n):
        v = A[:, i].copy()
        for j in range(i):
            R[j, i] = producto_escalar(Q[:, j], A[:, i])
            # restamos la proyección sobre q_j
            v -= R[j, i] * Q[:, j]
            nops += 2 * m + 1
        norma_v = norma(v, 2)
        nops += 2 * m

        if norma_v < tol:
            R[i, i] = 0
            Q[:, i] = np.zeros(m)
        else:
            R[i, i] = norma_v
            Q[:, i] = v / norma_v
            nops += m

    if retorna_nops:
        return Q, R, nops
    else:
        return Q, R


def calculaLDV(A):
    """
    Factorización A = L D V (L triangular inferior con 1s en la diagonal,
    D diagonal, V triangular superior con 1s en la diagonal), sin pivoteo.
    Devuelve None si A no es cuadrada o si aparece un pivote (casi) nulo.
    """
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
    """True si A es simétrica definida positiva (simétrica y con D > 0 en su LDV)."""
    if A is None:
        return False

    # simetría
    n = len(A)
    for i in range(n):
        for j in range(n):
            if abs(A[i][j] - A[j][i]) > atol:
                return False

    ldv = calculaLDV(A)
    if ldv is None:
        return False

    _, D, _ = ldv

    # todos los pivotes positivos
    for i in range(n):
        if D[i, i] <= atol:
            return False

    return True


def calculaCholesky(A, atol=1e-10):
    """
    Devuelve R triangular inferior tal que A = R R^T, armada como R = L sqrt(D)
    a partir de A = L D L^T. Devuelve None si A no es SDP.
    """
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


#%% Módulo ALC - Autovalores y SVD

def metpot2k(A, tol=1e-13, K=10000):
    """
    Método de la potencia. Devuelve (autovector, autovalor dominante, iteraciones).
    En cada paso normaliza por la componente de mayor módulo.
    """
    K = int(K)
    n = A.shape[0]
    v = np.random.rand(n)
    v = v / norma(v, 2)
    L = 0
    for k in range(K):
        w = matvec(A, v)
        indice = componente_max(w)
        L = w[indice]
        if L == 0:
            return v, 0.0, k
        v_nuevo = w / L
        if norma(v_nuevo - v, 2) < tol:
            return v_nuevo, L, k + 1
        v = v_nuevo
    return v, L, K


def diagRH(A, tol=1e-15, K=1e5):
    """
    Diagonaliza A simétrica con reflexiones de Householder: A = S D S^T.
    Va sacando el autovalor dominante con el método de la potencia y deflaciona.
    (No contempla el caso 1x1.)
    """
    n = A.shape[0]
    v1, l1, _ = metpot2k(A, tol, K)
    v1 = v1 / norma(v1, 2)

    # reflexión de Householder H que lleva v1 a e1
    e1 = np.zeros(n)
    e1[0] = 1
    u = e1 - v1
    nu = norma(u, 2)
    if nu < 1e-14:
        H = np.eye(n)
    else:
        H = np.eye(n) - 2 * producto_interno(u, u) / nu**2

    if n == 2:
        S = H
        D = multiplicacion_matrices(multiplicacion_matrices(H, A), H.T)
    else:
        # H A H^T tiene l1 en la esquina; seguimos con el bloque restante
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


def svd_reducida(A, k="max", tol=1e-15):
    """
    A la matriz de interés (de m x n)
    k el número de valores singulares (y vectores) a retener.
    tol la tolerancia para considerar un valor singular igual a cero
    Retorna U_red (matriz de m x k), valores_sing (vector de k valores singulares) y V_red (matriz de n x k)
    """
    A = np.array(A, dtype=float)
    m, n = A.shape

    # si m < n trabajamos con A^T
    traspuesta = m < n
    if traspuesta:
        A = transpuesta(A)
        m, n = n, m

    # autovectores de A^T A = columnas de V, autovalores = valores singulares al cuadrado
    AtA = multiplicacion_matrices(transpuesta(A), A)
    if n == 1:
        S, D = np.ones((1, 1)), AtA  # diagRH no contempla el caso 1x1
    else:
        S, D = diagRH(AtA)
    autovalores = [D[i, i] for i in range(n)]

    # diagRH devuelve primero los autovalores no nulos, de mayor a menor
    # (los asserts fallan si no se cumple)
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


#%% Ejercicio 2 - Calibración con objeto 3D

def obtenerQ3d(imgPts, objPts):
    """
    Arma la matriz Q (2N x 12) del sistema Q m = 0 del DLT 3D.
    Cada correspondencia (u, v) <-> P = (X, Y, Z, 1) aporta dos filas:
        [ P   0  -u P ]
        [ 0   P  -v P ]
    """
    N = len(objPts)
    Q = np.zeros((2 * N, 12))
    for i in range(N):
        u, v = imgPts[i][0], imgPts[i][1]
        P = [objPts[i][0], objPts[i][1], objPts[i][2], 1.0]
        for k in range(4):
            # primera fila: columnas 0-3 -> P, 4-7 -> 0, 8-11 -> -u P
            Q[2*i, k] = P[k]
            Q[2*i, 4 + k] = 0.0
            Q[2*i, 8 + k] = -u * P[k]

            # segunda fila: columnas 0-3 -> 0, 4-7 -> P, 8-11 -> -v P
            Q[2*i + 1, k] = 0.0
            Q[2*i + 1, 4 + k] = P[k]
            Q[2*i + 1, 8 + k] = -v * P[k]
    return Q


def resolverDLT3D(Q):
    """
    Busca m con ||m|| = 1 que minimiza ||Q m||: el autovector de Q^T Q asociado
    al menor autovalor (vector singular derecho de Q del menor valor singular).
    Devuelve m reacomodado como la matriz de proyección M de 3x4.
    """
    A = multiplicacion_matrices(transpuesta(Q), Q)
    S, D = diagRH(A)
    m = S[:, -1]  # última columna de S (menor autovalor)
    m = m / norma(m, 2)
    return m.reshape(3, 4)


def espejar(A, filas):
    """Invierte el orden de las filas (P A) o de las columnas (A P), con P la identidad espejada."""
    if (filas):
        return A[::-1]
    else:
        return A[:, ::-1]


def devolverMatricesCalibracion3d(M, restric=None):
    """
    Descompone M = K [R | t] con K triangular superior y R ortogonal.

    restric puede ser:
        None         -> sin restricciones
        'skew'       -> skew = 0
        'alfa'       -> alfa_u = alfa_v
        'skew-alfa'  -> las dos cosas
    Devuelve (R, t, K), con K normalizada para que K[2][2] = 1.
    """
    if (np.shape(M) != (3, 4)):
        print('M no cumple el tamaño de la matriz de proyeccion pedida')
        return None, None, None

    if (M[2][3] < 0):  # corregimos el signo
        M = -M

    Mprima = M[:, :3]
    MprimaT = transpuesta(Mprima)
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    Q, R = QR_con_GS(multiplicacion_matrices(MprimaT, P))  # QR de M'^T P

    # K = P R^T P (espejar equivale a multiplicar por P, sin hacer el producto)
    Kres = espejar(espejar(transpuesta(R), True), False).copy()
    # R = P Q^T
    Rres = espejar(transpuesta(Q), True).copy()

    if restric is not None:
        if restric in ('skew', 'skew-alfa'):  # skew = 0
            Kres[0][1] = 0.0
        if restric in ('alfa', 'skew-alfa'):  # alfa_u = alfa_v (promedio de los dos)
            Kres[0][0] = (Kres[0][0] + Kres[1][1]) / 2
            Kres[1][1] = Kres[0][0]

        # recalculamos R con el nuevo K para que K R = M'
        Rres = np.zeros((3, 3))
        for j in range(3):
            Rres[:, j] = res_tri(Kres, Mprima[:, j], inferior=False)

        # ortonormalizamos R (Gram-Schmidt sobre las filas, desde la última).
        # Ojo: después de esto K R aproxima M', ya no es exacto.
        r3 = Rres[2] / norma(Rres[2], 2)
        r2 = Rres[1] - producto_escalar(Rres[1], r3)*r3
        r2 = r2 / norma(r2, 2)
        r1 = Rres[0] - producto_escalar(Rres[0], r3)*r3 - producto_escalar(Rres[0], r2)*r2
        r1 = r1 / norma(r1, 2)
        Rres = np.array([r1, r2, r3])

    # t sale de resolver K t = (cuarta columna de M)
    t = res_tri(Kres, M[:, 3], inferior=False)

    # normalizamos K para que K[2][2] = 1
    Kres = Kres / Kres[2][2]

    return Rres, t, Kres


# --- Aplicación a los datos del ejercicio 2 ---

# Puntos del mundo real (U, V, W), ¿en cm? -> matriz de 12 x 3
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

# Cada fila del archivo es un cuadro con 24 números: u1 v1 u2 v2 ... u12 v12,
# en el mismo orden que objPts (u1 v1 <-> [0, 8, 0], etc.).
ruta_datos = os.path.join("calib", "3d", "detected_corners.txt")
datos = np.loadtxt(ruta_datos)


def imgpts_del_cuadro(k):
    """Devuelve los 12 puntos (u, v) del cuadro k como matriz de 12 x 2."""
    fila = datos[k]
    imgPts = np.zeros((12, 2))
    for i in range(12):
        imgPts[i, 0] = fila[2*i]      # u_i
        imgPts[i, 1] = fila[2*i + 1]  # v_i
    return imgPts


imgPts = imgpts_del_cuadro(0)
Q = obtenerQ3d(imgPts, objPts)
M = resolverDLT3D(Q)
R, T, K = devolverMatricesCalibracion3d(M)


#%% Ejercicio 3 - Calibración con patrón plano

def obtenerQ2d(imgPts, objPts):
    """
    Arma la matriz Q (2N x 9) del sistema Q h = 0 de la homografía.
    Cada correspondencia (u, v) <-> P = (X, Y, 1) aporta dos filas:
        [ P   0  -u P ]
        [ 0   P  -v P ]
    """
    N = len(objPts)
    Q = np.zeros((2 * N, 9))
    for i in range(N):
        u, v = imgPts[i][0], imgPts[i][1]
        P = [objPts[i][0], objPts[i][1], 1.0]
        for k in range(3):
            # primera fila: columnas 0-2 -> P, 3-5 -> 0, 6-8 -> -u P
            Q[2*i, k] = P[k]
            Q[2*i, 3 + k] = 0.0
            Q[2*i, 6 + k] = -u * P[k]

            # segunda fila: columnas 0-2 -> 0, 3-5 -> P, 6-8 -> -v P
            Q[2*i + 1, k] = 0.0
            Q[2*i + 1, 3 + k] = P[k]
            Q[2*i + 1, 6 + k] = -v * P[k]
    return Q


def v(H, i, j):
    """Vector v_ij armado con las columnas i y j de H (método de Zhang)."""
    hi, hj = H[:, i], H[:, j]
    return np.array([
        hi[0]*hj[0],                # ci1*cj1
        hi[0]*hj[1] + hi[1]*hj[0],  # ci1*cj2 + ci2*cj1
        hi[1]*hj[1],
        hi[0]*hj[2] + hi[2]*hj[0],
        hi[1]*hj[2] + hi[2]*hj[1],
        hi[2]*hj[2],
    ])


def armarA(Hlist):
    """Apila, por cada homografía, las filas v_01 y v_00 - v_11 en A (2N x 6)."""
    N = len(Hlist)
    A = np.zeros((2*N, 6))
    for i in range(N):
        H = Hlist[i]
        A[2*i] = v(H, 0, 1)
        A[2*i + 1] = v(H, 0, 0) - v(H, 1, 1)
    return A


def calcularMatrizK2D(A):
    """
    Calcula K a partir de A b = 0, con B = K^{-T} K^{-1} (simétrica).
    Devuelve K normalizada para que K[2,2] = 1, o None si B no resulta SDP.
    """
    # b con ||b|| = 1 que minimiza ||A b||: autovector de A^T A del menor autovalor
    AtA = multiplicacion_matrices(transpuesta(A), A)
    S, D = diagRH(AtA)
    b = S[:, -1]
    b = b / norma(b, 2)

    B = np.array([[b[0], b[1], b[3]], [b[1], b[2], b[4]], [b[3], b[4], b[5]]])
    if B[0, 0] < 0:
        B = -B  # b está definido a menos de un signo; lo queremos positivo para Cholesky

    # B = L L^T y B = K^{-T} K^{-1}  =>  K = (L^T)^{-1}
    L = calculaCholesky(B)
    if L is None:
        return None
    K = matriz_inversa(transpuesta(L))
    # K sale a menos de una escala; dividimos por K[2,2] para que quede en 1
    return K / K[2, 2]


def resolverDLT2D(Q):
    """
    Busca h con ||h|| = 1 que minimiza ||Q h|| (autovector de Q^T Q del menor
    autovalor) y lo devuelve reacomodado como la homografía H de 3x3.
    """
    A = multiplicacion_matrices(transpuesta(Q), Q)
    S, D = diagRH(A)
    h = S[:, -1]  # última columna de S (menor autovalor)
    h = h / norma(h, 2)
    return h.reshape(3, 3)


def obtenerPose2D(Hj, K):
    """
    Obtiene la pose (R, t) de una imagen a partir de su homografía Hj y de K.
    Viene de: lambda (u, v, 1)^T = K [r1 r2 t] (U, V, 1)^T = H (U, V, 1)^T
    """
    K_inv = matriz_inversa(K)
    c1, c2, c3 = Hj[:, 0], Hj[:, 1], Hj[:, 2]

    landa = 1 / norma(matvec(K_inv, c1), 2)
    r1 = landa * (matvec(K_inv, c1))
    r2 = landa * (matvec(K_inv, c2))
    t = landa * (matvec(K_inv, c3))

    # el tablero tiene que quedar adelante de la cámara: tz > 0
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t

    r3 = producto_cruz(r1, r2)  # garantiza ortogonalidad

    # armamos R = [r1 r2 r3]
    R = np.zeros((3, 3))
    R[:, 0], R[:, 1], R[:, 2] = r1, r2, r3

    # por el ruido R puede no quedar ortogonal: ortonormalizamos con GS (nos quedamos con la Q)
    resultado = QR_con_GS(R)
    if resultado is not None:
        R, _ = resultado

    return R, t


"""
# --- Aplicación a los datos del ejercicio 3 ---

# objPts del tablero: esquinas (30*i, 30*j), i = 0..10, j = 0..6
objPts = []
for j in range(7):       # filas (esquinas internas)
    for i in range(11):  # columnas
        objPts.append([30.0*i, 30.0*j, 0.0])

objPts = np.array(objPts)

# DataFrame con las esquinas marcadas
ruta_esquinas = os.path.join("calib", "coplanar", "marcadas", "esquinas.csv")
imagenes = pd.read_csv(ruta_esquinas)

# una homografía por imagen
# (en el notebook no usamos unique porque creo que no está permitido; si está, modificarlo)
Hlist = []
for nombre in imagenes["imagen"].unique():
    d = imagenes[imagenes["imagen"] == nombre].sort_values('punto')
    imgPts = d[["u", "v"]].to_numpy()

    Q = obtenerQ2d(imgPts, objPts)
    H = resolverDLT2D(Q)
    Hlist.append(H)

# A con todas las homografías
A = armarA(Hlist)

# matriz de parámetros intrínsecos K
K = calcularMatrizK2D(A)

# pose (R, t) de cada imagen del tablero
poses = []
for i, H in enumerate(Hlist):
    R, t = obtenerPose2D(H, K)
    poses.append((R, t))
"""
