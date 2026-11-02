from alc import producto_escalar,norma, producto_cruz,multiplicacion_matrices,matvec, transpuesta, matriz_inversa
import numpy as np
import os

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
    # CORRECCIÓN by Lu:
    # Cambié las dimensiones de t y M de (3,1) a (3,), porque
    # matvec de alc.py trabaja con vectores unidimensionales.
    # Antes, al pasarle una matriz columna, producto_escalar devolvía
    # un arreglo en lugar de un escalar, generando un ValueError.
    # También se modificaron los índices de Pc y p para adaptarlos
    # a los vectores unidimensionales.
    # El procedimiento matemático de la función es el mismo, revisen.

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
    # CORRECCIONES by Lu:
    # 1. Reemplacé n por N al actualizar eps, ya que N representa
    #    la cantidad de puntos y n es el vector normal al plano (asumo que era 
    #    un error de tipeo).
    # 2. Corregí las condiciones que seleccionan la mejor
    #    pose candidata y la mejor pose entre iteraciones. (tenían errores)
    # 3. Corregí la condición de convergencia: ahora se verifica
    #    que rms_prev no sea None y se usa break en lugar de
    #    return None para conservar la mejor pose encontrada.
    #    Si el RMS empeora o mejora menos que atol, se detiene.
    # 4. Agregué verificaciones para datos degenerados o dimensiones incorrectas.
    # 5. Se preserva el esquema POSIT del enunciado: POS coplanar,
    #    dos poses candidatas, selección por RMS y actualización de eps.

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

# APLICACION
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
        
    
# CORRECCIONES EN LA APLICACION:
# - Se agregan los argumentos faltantes en evaluarPose y reproyectar_puntos.
# - La altura se calcula a partir del centro de cámara C = -R^T t, no de t[2].
# - La traslación deseada se obtiene de t_des = -R_des*C_des (ecuación 15).
# - La imagen original solo sirve para evaluar la pose actual, no la deseada.
# - Se verifica que la versión importada de producto_cruz esté corregida.

# Evita ejecutar toda la aplicación cuando se importan funciones desde el notebook.
if __name__ == "__main__":
    # obj_pts, puntos en el mundo real
    objPts = np.array([
            [0.000,0.0,0.0], # centro
            [-0.175,0.3,0.0], # sup izquierda(donde esta el punto blanco)
            [0.175,0.3,0.0], # sup derecha
            [0.175,-0.3,0.0], # inferior derecha
            [-0.175,-0.3,0.0] # inferior izquierda
    ])

    # puntos de la imagen(los que marcamos) mismo orden que los otros puntos
    imgPts = np.array([
        [524,1002],
        [400,1097],
        [389,941],
        [644,911],
        [660,1062]], dtype=float)

    # Comprobamos que alc.py tiene corregido el producto cruz.
    if producto_cruz(np.array([1.0,0.0,0.0]), np.array([0.0,0.0,1.0]))[1] != -1:
        raise ValueError("Corregir producto_cruz en alc.py antes de aplicar POSIT")

    K = np.loadtxt(os.path.join("heli", "K.csv"),delimiter=";")
    # como se ve la camara ahora
    pose = aplicarPOSIT(objPts,imgPts,K,MaxIt=50,atol=1e-4)
    if pose is None:
        raise ValueError("POSIT no encontró una pose válida con estos puntos")
    R,t = pose
    print("Rotación actual R:\n", R)
    print("Traslación actual t:\n", t)
    print("Puntos válidos y RMS actuales:", evaluarPose(R,t,objPts,imgPts,K))

    # para orientar la camara verticalmente alineamos los ejes de la camara con los del mundo
    # U -> + x(ambos estan a la derecha)
    # V apunta hacia arriba, y hacia abajo, entonces V -> -y
    # Z apunta hacia el suelo y W apunta hacia el cielo(dron), ambos en direcciones opuestas entonces, W -> -z
    # queda R deseada conformada
    R_des = np.array([[1,0,0],
                     [0,-1,0],
                     [0,0,-1]], dtype=float)

    # centro actual de la camara, expresado en coordenadas del mundo (ecuación 15)
    C = -np.array(matvec(transpuesta(R),t),dtype=float).reshape(3)

    # usamos misma altura que la de ahora
    Z_des = C[2]  # t[2] no equivale a la altura cuando R no es vertical
    if Z_des <= 0:
        raise ValueError("La pose calculada ubica la cámara debajo del plano de la H")
    C_des = np.array([0.0,0.0,Z_des]) # centro de la cámara sobre la H
    t_des = -np.array(matvec(R_des,C_des),dtype=float).reshape(3)

    # rotacion que lleva de la actual a la deseada
    R_corregida = multiplicacion_matrices(R_des,transpuesta(R))

    dC = C_des - C # desplazamiento en el sistema de la h
    d_cam = np.array(matvec(R,dC), dtype=float).reshape(3) # desplazamiento en los ejes de la camara
    imgPts_deseados = reproyectar_puntos(R_des,t_des,objPts,K)

    # verificamos los puntos clave
    print("\nCentro de la H proyectado(debe ser cx,cy de K):")
    print("-> u,v = ", imgPts_deseados[0])
    print("-> cx,cy=", K[0,2], K[1,2])

    print("\nEsquina superior izquierda(punto blanco):")
    print("-> u,v =", imgPts_deseados[1])

    print("\nCentro actual C:", C)
    print("Centro deseado C_des:", C_des)
    print("Rotación deseada R_des:\n", R_des)
    print("Traslación deseada t_des:", t_des)
    print("Rotación de corrección R_corregida:\n", R_corregida)
    print("Desplazamiento expresado en ejes actuales d_cam:", d_cam)

    # No usar evaluarPose(R_des,t_des,objPts,imgPts,K): los píxeles imgPts
    # pertenecen a la imagen original, no a la imagen que tomaría el dron tras moverse.
