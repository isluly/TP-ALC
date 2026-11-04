"""
Genera salguero/marcados.csv con la posición en píxeles (u, v) del taxi, la camioneta
y la señora de buzo rojo en las capturas de salguero/gs_dataset (ejercicio 5c).

Uso (desde la carpeta del TP, en una terminal, NO en Jupyter inline):
    python marcar_salguero.py                 # marca los tres agentes
    python marcar_salguero.py senora          # marca solo uno (taxi, camioneta o senora)

Cómo se marca:
  - Al mover el mouse sobre la imagen aparece una LUPA con una cruz: el centro de la cruz
    es el píxel que se va a guardar. Un clic izquierdo marca el punto y pasa a la siguiente.
  - Ruedita del mouse (o + / -): más o menos aumento en la lupa.
  - Enter: "no se ve en esta imagen", se saltea.
  - Backspace: vuelve a la imagen anterior para rehacer ese punto.
  - Cerrar la ventana corta el marcado; lo ya marcado queda guardado.
  (Si activás el zoom o el pan de la barra de herramientas, los clics no marcan;
   desactivalos para seguir marcando.)

Se puede cortar y retomar: si el CSV ya existe, saltea los (imagen, agente) ya marcados.

IMPORTANTE: marcar SIEMPRE un punto sobre el piso (Z = 0), porque la homografía
solo vale para el plano de la calle, y SIEMPRE el mismo punto físico en todas las imágenes.

Solo para preparar datos: usa eventos de matplotlib, NO va en alc.py.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CARPETA = os.path.join("salguero", "gs_dataset")
SALIDA = os.path.join("salguero", "marcados.csv")
PASO = 3            # marcar una de cada PASO imágenes (siempre incluye la primera y la última)
LUPA_RADIO = 25     # medio lado (en píxeles de la imagen) de lo que muestra la lupa
LUPA_TAMANO = 0.28  # lado de la lupa, como fracción del ancho de la ventana

# qué punto marcar de cada agente (el mismo en todas las imágenes)
AGENTES = {
    "taxi":      "TAXI (negro, techo amarillo; al principio es el auto de adelante, lejos): "
                 "donde la rueda delantera de la izquierda de la imagen toca el piso",
    "camioneta": "CAMIONETA (pickup azul, viene detrás del taxi): "
                 "donde la rueda delantera de la izquierda de la imagen toca el piso",
    "senora":    "SEÑORA de rojo: punto del piso entre sus dos pies, a la altura del pie apoyado",
}
COLORES = {"taxi": "yellow", "camioneta": "cyan", "senora": "red"}


def tiempo(nombre):
    """img_10_18_47_351.jpg -> segundos desde las 00:00 (10*3600 + 18*60 + 47 + 0.351)."""
    hh, mm, ss, ms = nombre[len("img_"):-len(".jpg")].split("_")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def elegir_imagenes():
    """Lista de capturas ordenada por tiempo, tomando una de cada PASO (+ la última)."""
    todas = sorted([f for f in os.listdir(CARPETA) if f.startswith("img_") and f.endswith(".jpg")],
                   key=tiempo)
    elegidas = todas[::PASO]
    if elegidas[-1] != todas[-1]:
        elegidas.append(todas[-1])
    return elegidas


def guardar(filas):
    df = pd.DataFrame(filas, columns=["imagen", "t", "agente", "u", "v"])
    df = df.sort_values(["agente", "t"])
    df.to_csv(SALIDA, index=False)


class Marcador:
    """Ventana con la imagen y una lupa que sigue al mouse. Cada clic o tecla llama a al_responder."""

    def __init__(self, al_responder):
        self.al_responder = al_responder  # función que recibe (u, v), None o 'atras'
        self.fig, self.ax = plt.subplots(figsize=(14, 10.5))
        self.fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.9)
        self.radio = LUPA_RADIO

        # lupa: ejes chicos que se mueven con el mouse
        self.ax_lupa = self.fig.add_axes([0, 0, LUPA_TAMANO, LUPA_TAMANO * 14 / 10.5])
        self.ax_lupa.set_xticks([]); self.ax_lupa.set_yticks([])
        for borde in self.ax_lupa.spines.values():
            borde.set_edgecolor("white"); borde.set_linewidth(2)
        self.img_lupa = self.ax_lupa.imshow(np.zeros((2, 2, 3)), interpolation="nearest",
                                            extent=(-0.5, 0.5, 0.5, -0.5))
        self.ax_lupa.axhline(0, color="lime", lw=0.8)
        self.ax_lupa.axvline(0, color="lime", lw=0.8)
        self.ax_lupa.set_xlim(-0.5, 0.5); self.ax_lupa.set_ylim(0.5, -0.5)
        self.ax_lupa.set_visible(False)

        # blitting: redibuja solo la lupa (fluido); si el backend no lo soporta, redibuja todo
        self.blit = getattr(self.fig.canvas, "supports_blit", False)
        self.ax_lupa.set_animated(self.blit)
        self.fondo = None

        c = self.fig.canvas
        c.mpl_connect("draw_event", self._al_dibujar)
        c.mpl_connect("motion_notify_event", self._al_mover)
        c.mpl_connect("button_press_event", self._al_clic)
        c.mpl_connect("key_press_event", self._al_tecla)
        c.mpl_connect("scroll_event", self._al_scroll)
        c.mpl_connect("close_event", self._al_cerrar)
        self.cerrada = False
        self.ultimo_mouse = None

    # ---- mostrar una imagen ----
    def mostrar(self, img, titulo, consigna, previos, color):
        self.img = img
        alto, ancho = img.shape[:2]
        r = 200  # padding para que la lupa funcione en los bordes
        self.img_pad = np.zeros((alto + 2 * r, ancho + 2 * r, img.shape[2]), dtype=img.dtype)
        self.img_pad[r:r + alto, r:r + ancho] = img
        self.pad = r
        self.ax.clear()
        self.ax.imshow(img)
        if len(previos) > 0:
            self.ax.plot(previos[:, 0], previos[:, 1], "o-", color=color, ms=4, lw=1)
        self.ax.set_xlim(0, ancho); self.ax.set_ylim(alto, 0)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.fig.suptitle(f"{titulo}\n{consigna}\n"
                          "clic = marcar · Enter = no se ve · Backspace = volver · ruedita = aumento",
                          fontsize=10)
        self.fig.canvas.draw_idle()
        if self.ultimo_mouse is not None:
            self._actualizar_lupa(*self.ultimo_mouse)

    # ---- respuesta: se le pasa al manejador, que decide qué mostrar después ----
    def _responder(self, r):
        self.al_responder(r)

    # ---- lupa ----
    def _al_dibujar(self, ev):
        if self.blit:
            self.fondo = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _actualizar_lupa(self, u, v, x_pant=None, y_pant=None):
        r, p = self.radio, self.pad
        ui, vi = int(round(u)), int(round(v))
        recorte = self.img_pad[vi + p - r: vi + p + r + 1, ui + p - r: ui + p + r + 1]
        self.img_lupa.set_data(recorte)
        # cada píxel del recorte ocupa 1/(2r+1) del eje; se centra en el punto exacto (subpíxel)
        n = 2 * r + 1
        du, dv = (ui - u) / n, (vi - v) / n
        self.img_lupa.set_extent((-0.5 + du, 0.5 + du, 0.5 + dv, -0.5 + dv))

        # ubicar la lupa arriba a la derecha del cursor (o del otro lado si no entra)
        if x_pant is not None:
            W, H = self.fig.bbox.width, self.fig.bbox.height
            ancho_l, alto_l = LUPA_TAMANO, LUPA_TAMANO * W / H
            fx, fy = x_pant / W, y_pant / H
            x0 = fx + 0.03 if fx + 0.03 + ancho_l < 1 else fx - 0.03 - ancho_l
            y0 = fy + 0.03 if fy + 0.03 + alto_l < 1 else fy - 0.03 - alto_l
            self.ax_lupa.set_position([x0, y0, ancho_l, alto_l])
            self.ultimo_pant = (x_pant, y_pant)
        self.ax_lupa.set_visible(True)
        self._redibujar_lupa()

    def _ocultar_lupa(self):
        if self.ax_lupa.get_visible():
            self.ax_lupa.set_visible(False)
            self._redibujar_lupa()

    def _redibujar_lupa(self):
        c = self.fig.canvas
        if self.blit and self.fondo is not None:
            c.restore_region(self.fondo)
            if self.ax_lupa.get_visible():
                self.fig.draw_artist(self.ax_lupa)
            c.blit(self.fig.bbox)
        else:
            c.draw_idle()

    # ---- eventos ----
    def _herramienta_activa(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return tb is not None and str(getattr(tb, "mode", "")) != ""

    def _al_mover(self, ev):
        if ev.inaxes is self.ax and ev.xdata is not None and not self._herramienta_activa():
            self.ultimo_mouse = (ev.xdata, ev.ydata, ev.x, ev.y)
            self._actualizar_lupa(ev.xdata, ev.ydata, ev.x, ev.y)
        else:
            self._ocultar_lupa()

    def _al_clic(self, ev):
        if ev.inaxes is self.ax and ev.button == 1 and ev.xdata is not None \
                and not self._herramienta_activa():
            self._responder((ev.xdata, ev.ydata))

    def _al_tecla(self, ev):
        if ev.key in ("enter", "escape"):
            self._responder(None)
        elif ev.key == "backspace":
            self._responder("atras")
        elif ev.key in ("+", "="):
            self._cambiar_aumento(-1)
        elif ev.key == "-":
            self._cambiar_aumento(+1)

    def _al_scroll(self, ev):
        self._cambiar_aumento(-1 if ev.button == "up" else +1)

    def _cambiar_aumento(self, signo):
        self.radio = int(min(120, max(6, round(self.radio * (1.25 if signo > 0 else 0.8)))))
        if self.ultimo_mouse is not None:
            self._actualizar_lupa(*self.ultimo_mouse)

    def _al_cerrar(self, ev):
        self.cerrada = True


def main():
    agentes = sys.argv[1:] if len(sys.argv) > 1 else list(AGENTES)
    for a in agentes:
        if a not in AGENTES:
            print(f"agente desconocido: {a} (opciones: {', '.join(AGENTES)})"); return

    # retomar lo ya marcado
    filas = []
    if os.path.exists(SALIDA):
        filas = pd.read_csv(SALIDA).values.tolist()
        print(f"{SALIDA}: {len(filas)} puntos ya marcados, se saltean")
    hechos = {(f[0], f[2]) for f in filas}

    imagenes = elegir_imagenes()
    tareas = [(a, k, n) for a in agentes for k, n in enumerate(imagenes) if (n, a) not in hechos]
    print(f"{len(imagenes)} imágenes por agente (PASO = {PASO}), {len(tareas)} puntos por marcar")
    if not tareas:
        print("No queda nada por marcar."); return

    cache = {}
    estado = {"i": 0}

    def mostrar_actual():
        agente, k, nombre = tareas[estado["i"]]
        if nombre not in cache:
            cache[nombre] = mpimg.imread(os.path.join(CARPETA, nombre))
        previos = np.array([[f[3], f[4]] for f in filas if f[2] == agente]).reshape(-1, 2)
        m.mostrar(cache[nombre], f"{nombre}  —  {agente} {k + 1}/{len(imagenes)}",
                  AGENTES[agente], previos, COLORES[agente])

    def al_responder(r):
        """Se llama con cada clic / tecla: guarda, avanza o retrocede y muestra la próxima."""
        agente, k, nombre = tareas[estado["i"]]
        if r == "atras":
            if estado["i"] == 0:
                return
            estado["i"] -= 1
            a_prev, _, n_prev = tareas[estado["i"]]
            filas[:] = [f for f in filas if not (f[0] == n_prev and f[2] == a_prev)]
            guardar(filas)
            print(f"volviendo a {n_prev} {a_prev}")
            mostrar_actual()
            return
        if r is None:
            print(f"{nombre} {agente}: no se ve, salteada")
        else:
            u, v = float(r[0]), float(r[1])
            filas.append([nombre, tiempo(nombre), agente, round(u, 1), round(v, 1)])
            guardar(filas)  # se guarda en cada clic, así cortar no pierde nada
            print(f"{nombre} {agente}: u={u:.1f} v={v:.1f}")
        estado["i"] += 1
        if estado["i"] < len(tareas):
            mostrar_actual()
        else:
            print("Listo, no queda nada por marcar.")
            plt.close(m.fig)

    m = Marcador(al_responder)
    mostrar_actual()
    plt.show()  # bloquea hasta que se cierra la ventana (o se termina de marcar)

    guardar(filas)
    print(f"Guardado: {len(filas)} puntos en {SALIDA}")


if __name__ == "__main__":
    main()
