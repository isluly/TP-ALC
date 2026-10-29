from alc import producto_escalar,norma, producto_cruz,multiplicacion_matrices,matvec, transpuesta, matriz_inversa
import numpy as np
import cmath

# sale de las ecuaciones 16 y 17 de la parte teorica(pag 19)
def calculoLambdaMu(I0,J0):
    re = producto_escalar(J0,J0) - producto_escalar(I0,I0)
    im = -2 * producto_escalar(I0,J0)
    C = cmath.sqrt(complex(re,im))
    return C.real, C.imag   # la otra solucion es -landa -mu que no la queremos

# toma una solucion candidata Is y Js y la convierte en una pose completa 
def calcularSalidaPOSIT(Is,Js,fx,m0,M0):
    nI,nJ = norma(Is,2), norma(Js,2)
    Z0 = 2 * fx / (nI + nJ) #promedio de ambas estimaciones, ya sabiendo que Z0 = f / ||I|| y Z0 = f/ ||J||
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
def evaluarPose(R,t,obj_vecs,img_vecs,K):
    validas = 0 # cuenta cuantos puntos quedan adelante de la camara
    suma = 0.0  # va acumulando los errores
    
    # convertimos t en 3x1 para poder sumarla a R*M
    t = np.array(t,dtype=float).reshape(3,1)
    
    # recorre los puntos de a pares, M = [U,V,W] y m es su pixel medido(u,v). Se corresponde el primero con el primero y el segundo con el segundo....
    for M, m in zip(obj_vecs,img_vecs):
        # redimensiono
        M = np.array(M, dtype=float).reshape(3,1)
        # Punto de la camara = R*M + t(donde esta ese punto visto desde la camara)
        Pc = matvec(R, M) + t

        # si el punto está atras de la cámara
        if Pc[2,0] <= 0:
            continue

        # proyeccion p = K*Pc y despues se divide por la tercera componente, x = fx/Z(se divide por la profundidad por efecto de cuanto mas lejos esta el objeto mas chico se ve en la camara)
        p = matvec(K,Pc)
        u = p[0,0] / p[2,0]
        v = p[1,0] / p[2,0]

        # compara lo predicho(pixeles) con lo medido m[0] y m[1] y suma el cuadrado de la distancia
        suma += (u - m[0])**2 + (v - m[1])**2
        # cuenta el punto como valido
        validas += 1

    # raiz del promedio de los errores al cuadrado
    rms = np.sqrt(suma / validas) if validas > 0 else None # devuelve infinito
    return validas,rms
    
# resuelve Ax=b con cuadrados minimos, AtA.x = At.b --> x = (AtA) a la menos 1 . At . b
# A tiene que estar ampliada con n si es coplanar, AtA es singular en ese caso
# se usa para resolver luego los sistemas
def resolver(A,b):
    AtA = multiplicacion_matrices(transpuesta(A), A)
    inv = matriz_inversa(AtA)
    Atb = matvec(transpuesta(A),b)
    # devuelve x
    return matvec(inv,Atb)


def aplicarPOSIT(obj_vecs,img_vecs,K,MaxIt=50,atol=1e-4):
    obj = np.array(obj_vecs,dtype=float)
    img = np.array(img_vecs,dtype=float)
    K = np.array(K,dtype=float)
    fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
    N = len(obj)
    
    # pasa de pixeles a coordenadas de imagen centradas, xi = ui - cx, yi = vi - cy  fx/fy(escalando los ejes)
    # dejamos ambos ejes en la misma escala con el factor fx/fy, si fx=fy no cambia nada
    m = np.zeros((N,2))
    for i in range(n):
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
        
        I0 = np.array(resolver(A_ampliada,bx),dtype=float).reshape(3)
        J0 = np.array(resolver(A_ampliada,by),dtype=float).reshape(3)
        
        lam,mu = calculoLambdaMu(I0,J0)
        
        # dos soluciones
        mejor_it = None
        for s in (1,-1):
            Is = I0 + s * lam * n
            Js = J0 + s * mu * n
            i,j,k,Z0,R,t = calcularSalidaPOSIT(Is,Js,fx,m0,M0)
            validas,rms = evaluarPose(R,t,obj,img,K)
            if validas == N and rms is not None: #aca se descartan poses con puntos atras de la camara
               if mejor_it is None or rms or mejor_it[0]:
                   mejor_it = (rms,R,np.array(t,dtype=float).reshape(3),k,Z0)
        
        if mejor_it is None:
            break
        
        rms,R,t,k,Z0 = mejor_it 
        
        if mejor is not None or rms < mejor[0]:
            mejor = (rms,R,t)
            
        # convergencia :  no mejora el rms entre iteraciones
        if abs(rms_prev - rms) < atol:
            return None
        
        rms_prev = rms
        
        # actualizar eps con la solucion elegida
        e = np.array(matvec(A,k), dtype = float).reshape(n-1)
        for i in range(n-1):
            eps[i] = e[i] / Z0
    
    if mejor is None:
        return None
    
    return mejor[1],mejor[2]

# APLICACION
# reproyecta puntos 3D a la imagen usando R,t y K
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

K = np.loadtxt(r"\heli\K.csv", delimiter = ",")
# como se ve la camara ahora
R,t = aplicarPOSIT(objPts,imgPts,K,MaxIt=50,atol=1e-4)
print(R,t,evaluarPose(R,t,objPts,imgPts))

# para girar la camara y que quede vertical(giro de 90% aprox) alineamos los ejes de la camara con los del mundo
# U -> + x(ambos estan a la derecha)
# V apunta hacia arriba, y hacia abajo, entonces V -> -y
# Z apunta hacia el suelo y W apunta hacia el cielo(dron), ambos en direcciones opuestas entonces, W -> -z
# queda R deseada conformada
R_des = np.array([[1,0,0],
                 [0,-1,0],
                 [0,0,-1]], dtype=float)

# usamos misma altura que la de ahora
Z_des = t[2]
t_des = np.array([0.0,0.0,Z_des]) # centro de la H sobre el eje optico

# rotacion que lleva de la actual a la deseada
R_corregida = multiplicacion_matrices(R_des,transpuesta(R))

# centro actuak
C = -np.array(matvec(transpuesta(R),t),dtype=float).reshape(3)
C_des = np.array([0.0,0.0,Z_des])

dC = C_des - C # desplazamiento en el sistema de la h
d_cam = np.array(matvec(R,dC), dtype=float).reshape(3) # desplazamiento en los ejes de la camara
imgPts_deseados = reproyectar_puntos(R_des,t_des,K)

# verificamos los puntos clave
print("Centro de la H proyectado(debe ser cx,cy de K):")
print("-> u,v = ", imgPts_deseados[0])
print("-> cx,cy=", K[0,2], K[1,2])

print("\nEsquina superior izquierda(punto blanco):")
print("-> u,v =", imgPts_deseados[1])

print(evaluarPose(R_des,t_des,objPts,imgPts,K)) # normal que el RMS sea grande