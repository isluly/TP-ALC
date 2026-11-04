import cv2
import numpy as np
import glob
import os
import csv
import time

CHECKERBOARD = (11, 7)
SQUARE = 30.0  # mm
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Puntos del mundo (U, V, 0), mismo orden que las esquinas de OpenCV
grid = np.zeros((77, 3), dtype=np.float64)
grid[:, :2] = np.dstack(np.meshgrid(np.arange(11) * SQUARE,
                                    np.arange(7) * SQUARE)).reshape(-1, 2)

OUT_DIR = 'calib/coplanar/marcadas'
os.makedirs(OUT_DIR, exist_ok=True)
images = sorted(glob.glob('calib/coplanar/Im_R_*.png'),
                key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split('_')[-1]))
total = len(images)

print(f"Tablero: {CHECKERBOARD[0]}x{CHECKERBOARD[1]} esquinas, casillero de {SQUARE:.0f} mm")
print(f"Imágenes encontradas: {total}")
if total == 0:
    print("[error] No se encontró ninguna imagen en calib/coplanar/Im_R_*.png")
    raise SystemExit(1)
print("-" * 50)

ok, fallidas = 0, []
t0 = time.time()

with open(os.path.join(OUT_DIR, 'esquinas.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['imagen', 'punto', 'U', 'V', 'Z', 'u', 'v'])

    for n, fname in enumerate(images, start=1):
        nombre = os.path.basename(fname)
        prefijo = f"[{n:>{len(str(total))}}/{total}] {nombre}"

        img = cv2.imread(fname)
        if img is None:
            print(f"{prefijo}: ERROR, no se pudo leer la imagen")
            fallidas.append(nombre)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        print(f"{prefijo}: buscando esquinas...", end=" ", flush=True)
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
        if not found:
            print("tablero NO detectado")
            fallidas.append(nombre)
            continue

        print("detectado, refinando subpíxel...", end=" ", flush=True)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        pts = corners.reshape(-1, 2)

        # Marcar cada esquina con un punto rojo (BGR) y guardar la imagen.
        # cv2.imwrite pisa el archivo si ya existe.
        for x, y in pts:
            cv2.circle(img, (int(round(x)), int(round(y))), 4, (0, 0, 255), -1)
        cv2.imwrite(os.path.join(OUT_DIR, nombre), img)

        # Una fila por esquina
        for k in range(77):
            w.writerow([nombre, k,
                        grid[k, 0], grid[k, 1], grid[k, 2],
                        f"{pts[k, 0]:.4f}", f"{pts[k, 1]:.4f}"])
        ok += 1
        print("listo (77 puntos guardados)")

print("-" * 50)
print(f"Procesadas correctamente: {ok}/{total}")
if fallidas:
    print(f"Fallidas ({len(fallidas)}): {', '.join(fallidas)}")
print(f"Imágenes marcadas en: {OUT_DIR}/")
print(f"CSV: esquinas.csv ({ok * 77} filas de datos)")
print(f"Tiempo total: {time.time() - t0:.1f} s")
