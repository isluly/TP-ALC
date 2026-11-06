"""
TP 2 - ÁLGEBRA LINEAL COMPUTACIONAL

Grupo: Triangulados
Integrantes: Posse Lucila, Ventroni Milena, Larreira Pilar, Krause Arnim Francisco 
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
    m, n = A.shape
    if (n != len(v)):
        print('No se puede')
        return None
    res = np.zeros(m)
    for i in range(m):
        res[i] = producto_escalar(A[i, :], v)  # fila i * v
    return res


def vecmat(v, A):
    """Producto vector por matriz: v^T A."""
    m, n = A.shape
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
        A = np.array(A, dtype=float)
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
    n, m = L.shape
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
        v = np.array(A[:, i], copy=True)
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


#%% Funciones ejercicio 2 - Calibración con objeto 3D

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
    if ( M.shape != (3, 4)):
        print('M no cumple el tamaño de la matriz de proyeccion pedida')
        return None, None, None

    if (M[2][3] < 0):  # corregimos el signo
        M = -M

    Mprima = M[:, :3]
    MprimaT = transpuesta(Mprima)
    P = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    Q, R = QR_con_GS(multiplicacion_matrices(MprimaT, P))  # QR de M'^T P

    # K = P R^T P (espejar equivale a multiplicar por P, sin hacer el producto)
    Kres = np.array(espejar(espejar(transpuesta(R), True), False), copy=True)
    # R = P Q^T
    Rres = np.array(espejar(transpuesta(Q), True), copy=True)

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


def imgpts_del_cuadro(k):
    """Devuelve los 12 puntos (u, v) del cuadro k como matriz de 12 x 2."""
    fila = datos[k]
    imgPts = np.zeros((12, 2))
    for i in range(12):
        imgPts[i, 0] = fila[2*i]      # u_i
        imgPts[i, 1] = fila[2*i + 1]  # v_i
    return imgPts

#%% Funciones ejercicio 3 - Calibración con patrón plano

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

#%% Funciones ejercicio 4 - Futbol

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

#%% Funciones ejercicio 5 - Salguero

def calleAImagen(H, X, Y):
    '''
    Proyecta un punto de la calle (Z = 0) a la imagen.
    H: homografía de la calle a la imagen. X, Y: coordenadas en metros.
    Devuelve el píxel (u, v).
    '''
    puntoHomogeneo = matvec(H, np.array([X, Y, 1]))
    return puntoHomogeneo[:2] / puntoHomogeneo[2]

def imagenACalle(Hinversa, pixel):
    '''
    Lleva un píxel al plano de la calle.
    Hinversa: inversa de la homografía de la calle a la imagen. pixel: array (u, v).
    Devuelve la posición (X, Y) en metros.
    '''
    puntoHomogeneo = matvec(Hinversa, np.array([pixel[0], pixel[1], 1]))
    return puntoHomogeneo[:2] / puntoHomogeneo[2]


def pendienteCuadradosMinimos(t, x):
    '''
    Devuelve la pendiente de la recta que ajusta x(t) por cuadrados mínimos.
    t, x: arrays de igual longitud.
    '''
    return np.sum((t - np.mean(t)) * (x - np.mean(x))) / np.sum((t - np.mean(t)) ** 2)


#%% Funciones ejercicio 6 -Dron

# sale de las ecuaciones 16 y 17 de la parte teorica(pag 19)
# Parámetros: I0,J0 son soluciones particulares del paso POS coplanar.
# Devuelve: una pareja lambda,mu que satisface las ecuaciones (16) y (17).
def calculoLambdaMu(I0,J0):
    re = producto_escalar(J0,J0) - producto_escalar(I0,I0)
    im = -2 * producto_escalar(I0,J0)
    C = complex(re, im) ** 0.5  # raíz cuadrada compleja sin importar cmath (no permitido en la entrega)
    return C.real, C.imag   # la otra solucion es -lambda, -mu; se prueba en aplicarPOSIT

# toma una solucion candidata Is y Js y la convierte en una pose completa 
# Parámetros: Is,Js son una solución candidata; fx es la focal común,
# m0 el píxel centrado del punto de referencia y M0 su posición 3D.
# Devuelve: los tres ejes, Z0, la rotación R y la traslación t.
def calcularSalidaPOSIT(Is,Js,fx,m0,M0):
    nI,nJ = norma(Is,2), norma(Js,2)
    if nI == 0 or nJ == 0:
        return None  # no se puede normalizar una solución nula
    Z0 = 2 * fx / (nI + nJ) # usa la igualdad ||I|| = ||J|| de la teoria: Z0 = f/||I|| = f/||J||
    i = Is / nI # normaliza para obetener los dos ejes de la camara, vectores unitarios
    j = Js / nJ
    k = producto_cruz(i,j) # sacamos el valor de k por producto cruz(tercer eje, perpendicular a los otros dos)
    R = np.array([i,j,k]) # filas i,j,k
    T = Z0 / fx * np.array([m0[0], m0[1],fx]) # T=OM0, posicion de M0 vista desde la camara. Sale de que M0 esta sobre el rayo que pasa por m0 a profundidad Z0
    t = T - (matvec(R,M0)) # pasa esa traslacion al origen. t es la posicion del origen del sistema H, en coordenadas de camara, si M0 es (0,0,0) entonces T=t
    return i,j,k,Z0,R,t

# mide que tan buena es una pose
# devuelve cantidad de proyecciones validas de obj_vecs y el RMS obtenido con respectco a las pos originales en img_vect
# cuantos puntos estaban adelante de la camara y el error promedio en pixeles(un RMS chico implica una pose buena)

# Parámetros: R,t definen la pose; obj_vecs/img_vecs son correspondencias;
# K es la matriz de parámetros intrínsecos.
# Devuelve: cantidad de puntos delante de cámara y RMS de reproyección.
def evaluarPose(R, t, obj_vecs, img_vecs, K):

    validas = 0 # cuenta cuantos puntos quedan adelante de la camara
    suma = 0.0  # va acumulando los errores

    # convertimos t en un vector de dimensión (3,) para poder sumarla a R*M
    t = np.array(t, dtype=float).reshape(3)

    # recorre los puntos de a pares, M = [U,V,W] y m es su pixel medido(u,v). Se corresponde el primero con el primero y el segundo con el segundo....
    for M, m in zip(obj_vecs, img_vecs):
        # redimensiono
        M = np.array(M, dtype=float).reshape(3)

        # Punto de la camara = R*M + t(donde esta ese punto visto desde la camara)
        Pc = matvec(R, M) + t

        # si el punto está atras de la cámara
        if Pc[2] <= 0:
            continue

        # proyeccion p = K*Pc y despues se divide por la tercera componente, x = fx/Z(se divide por la profundidad por efecto de cuanto mas lejos esta el objeto mas chico se ve en la camara)
        p = matvec(K, Pc)
        u = p[0] / p[2]
        v = p[1] / p[2]

        # compara lo predicho(pixeles) con lo medido m[0] y m[1] y suma el cuadrado de la distancia
        suma += (u - m[0])**2 + (v - m[1])**2

        # cuenta el punto como valido
        validas += 1

    # raiz del promedio de los errores al cuadrado
    rms = np.sqrt(suma / validas) if validas > 0 else None # devuelve None si no hay puntos válidos

    return validas, rms

    
# resuelve Ax=b con cuadrados minimos, AtA.x = At.b --> x = (AtA) a la menos 1 . At . b
# A tiene que estar ampliada con n si es coplanar, AtA es singular en ese caso
# se usa para resolver luego los sistemas
# Parámetros: A es la matriz del sistema lineal y b su lado derecho.
# Devuelve: solución de mínimos cuadrados, o None si AtA no es invertible.
def resolver(A,b):
    AtA = multiplicacion_matrices(transpuesta(A), A)
    inv = matriz_inversa(AtA)
    if inv is None:
        return None  # el sistema no pudo resolverse con la inversa implementada
    Atb = matvec(transpuesta(A),b)
    # devuelve x
    return matvec(inv,Atb)


# Parámetros: puntos 3D/2D, intrínsecos K, máximo de iteraciones y tolerancia RMS.
# Devuelve: (R,t) de la mejor pose factible o None si no se encuentra una.
def aplicarPOSIT(obj_vecs, img_vecs, K, MaxIt=50, atol=1e-4):

    obj = np.array(obj_vecs, dtype=float)
    img = np.array(img_vecs, dtype=float)
    K = np.array(K, dtype=float)
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    N = len(obj)
    # Se necesitan al menos 4 puntos, las correspondencias deben coincidir
    # y los parametros focales no pueden ser cero.
    if N < 4 or len(img) != N or obj.shape != (N, 3) or img.shape != (N, 2):
        return None
    if K.shape != (3, 3) or fx == 0 or fy == 0:
        return None

    # pasa de pixeles a coordenadas de imagen centradas, xi = ui - cx, yi = vi - cy  fx/fy(escalando los ejes)
    # dejamos ambos ejes en la misma escala con el factor fx/fy, si fx=fy no cambia nada
    m = np.zeros((N,2))
    for i in range(N):
        m[i,0] = img[i,0] - cx
        m[i,1] = (img[i,1] - cy) * fx / fy
    M0,m0 = obj[0], m[0]

    A = obj[1:] - M0
    # normal unitaria al plano, prodcto cruz, se elige el de mayor norma entre pares de filas, bien condicionado
    n,mejor_norma = None, 0.0
    for a in range(len(A)):
        for b in range(a + 1, len(A)):
            c = producto_cruz(A[a],A[b])
            nc = norma(c,2)
            if nc > mejor_norma:
                n,mejor_norma = np.array(c,dtype=float),nc

    # verificamos que sea posible calcular una normal al plano
    if mejor_norma == 0:
        return None

    # vector unitario
    n = n / mejor_norma

    # como los puntos son coplanares, AtA es singular, entonces se le agrega n.I0 = 0
    A_ampliada = np.zeros((N,3))
    for i in range(N-1):
        A_ampliada[i] = A[i]
    A_ampliada[N-1] = n

    eps = np.zeros(N-1)
    mejor = None
    rms_prev = None

    for iteracion in range(MaxIt):
        # lado derecho: x_i(1 + eps_i) - x0
        xp = m[1:,0] * (1 + eps) - m0[0]
        yp = m[1:,1] * (1 + eps) - m0[1]

        # ampliado con el 0 de la ecuacion
        bx = np.zeros(N)
        by = np.zeros(N)
        bx[:N-1] = xp
        by[:N-1] = yp

        I0_aux = resolver(A_ampliada,bx)
        J0_aux = resolver(A_ampliada,by)
        if I0_aux is None or J0_aux is None:
            break
        I0 = np.array(I0_aux,dtype=float).reshape(3)
        J0 = np.array(J0_aux,dtype=float).reshape(3)

        lam,mu = calculoLambdaMu(I0,J0)

        # dos soluciones
        mejor_it = None
        for s in (1,-1):
            Is = I0 + s * lam * n
            Js = J0 + s * mu * n
            salida = calcularSalidaPOSIT(Is,Js,fx,m0,M0)
            if salida is None:
                continue
            i,j,k,Z0,R,t = salida
            validas,rms = evaluarPose(R,t,obj,img,K)

            if validas == N and rms is not None: #aca se descartan poses con puntos atras de la camara
                if mejor_it is None or rms < mejor_it[0]:
                    mejor_it = (rms,R,np.array(t,dtype=float).reshape(3),k,Z0)

        if mejor_it is None:
            break

        rms,R,t,k,Z0 = mejor_it

        if mejor is None or rms < mejor[0]:
            mejor = (rms,R,t)

        # convergencia : no mejora el rms entre iteraciones
        if rms_prev is not None and rms_prev - rms < atol:
            break

        rms_prev = rms

        # actualizar eps con la solucion elegida
        e = np.array(matvec(A,k), dtype=float).reshape(N-1)
        for i in range(N-1):
            eps[i] = e[i] / Z0

    if mejor is None:
        return None

    return mejor[1],mejor[2]

# reproyecta puntos 3D a la imagen usando R,t y K
# Parámetros: pose R,t, puntos 3D y matriz K.
# Devuelve: coordenadas proyectadas en píxeles; None si quedan detrás.
def reproyectar_puntos(R,t,objPts,K):
    imgpts_proyectados = []
    t_vec = np.array(t,dtype=float).reshape(3)
    for pt in objPts:
        M = np.array(pt,dtype=float).reshape(3) # [X,Y,Z] del punto
        # aplicamos la rotacion al punto: R*M
        RM = matvec(R,M)
        # sumamos la traslacion
        Pc = RM + t_vec
        
        # verificamos que el punto este adelante de la camara
        if Pc[2] <= 0:
            imgpts_proyectados.append([None,None])
            continue
        
        # proyectamos con la matriz intrinseca K: p = K*Pc
        p = matvec(K,Pc)
        
        # pasamos a coordenadas de pixeles dividiendo por la profundidad
        u = p[0] / p[2]
        v = p[1] / p[2]
        
        imgpts_proyectados.append([u,v])
    return np.array(imgpts_proyectados, dtype=object)
        
