import numpy as np

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

