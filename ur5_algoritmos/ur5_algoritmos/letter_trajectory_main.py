#!/usr/bin/env python3
"""
letter_trajectory_main.py  —  V8 (escritura humana)

Nodo ROS2 que genera trayectorias a partir de una letra o palabra y las
ejecuta en el UR5 mediante control cinemático diferencial.

Flujo:
    1. Generar imagen de la palabra con PIL.
    2. Procesar imagen y extraer esqueleto.
    3. Extraer segmentos del grafo del esqueleto.
    4. Fusionar con lógica de escritura humana (izquierda a derecha, lazos completos,
       sin saltar letras, orden natural).
    5. Suavizar como splines.
    6. Escalar y mapear cada trayectoria al espacio cartesiano 3D del robot.
    7. Seguir cada punto con control cinemático (Jacobiano).
    8. Entre trayectorias: levantar el end-effector (pen-up).
"""

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

# Módulos del paquete UR5
from ur5_algoritmos.fk_functions import fkine_ur5, TF2xyzquat
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import compute_dq, pose_error
from ur5_algoritmos.markers import create_sphere_marker, set_marker_pose

# Funciones de trayectoria de letras (V8)
from ur5_algoritmos.letter_trajectory_functions import (
    plot_letter_pil,
    process_image,
    skeleton_to_points,
    analyze_skeleton_points,
    skeleton_to_ordered_segments,
    merge_segments_human_writing_order,
    segments_to_spline_trajectories,
)


# ===========================================================================
# Parámetros globales (ajustar según la celda del notebook)
# ===========================================================================

FONTS = {
    "1": ("Playwrite CU", "/home/utec/llm_ur5/V2/Tipografia1/PlaywriteCU-VariableFont_wght.ttf"),
}

# --- Texto y fuente ---
LETTER   = "Miguel"
FONT_KEY = "1"

# --- Parámetros de fusión (escritura humana) ---
MERGE_MAX_DIST           = 28.0    # px — sube si una letra queda partida
MERGE_MAX_ANGLE          = 125.0   # ° — sube si vuelta necesita cambio fuerte
MERGE_BRIDGE_PTS         = 5       # puntos interpolados en huecos pequeños
MERGE_CYCLE_DIST         = 24.0    # px — para abrir lazos cercanos
MERGE_EXPECTED_LETTER_GAP = None   # None = estimación automática

# --- Parámetros de spline ---
SPLINE_SMOOTHING = 1.5
SPLINE_NUM_PTS   = 200

# --- Mapeo al espacio cartesiano del robot ---
# El esqueleto vive en píxeles (imagen de 800×800).
# Se normaliza a [0, 1] y se escala a metros en el espacio del robot.
DRAW_CENTER  = np.array([0.40, 0.0, 0.45])   # centro del plano de escritura (m)
DRAW_WIDTH   = 0.20                            # ancho del área de dibujo (m)
DRAW_HEIGHT  = 0.10                            # alto del área de dibujo (m)
DRAW_Z       = 0.45                            # altura Z de contacto (m)
LIFTOFF_Z    = 0.55                            # altura Z de vuelo entre trazos (m)
IMAGE_SIZE   = 800                             # tamaño de la imagen generada

# --- Control cinemático ---
DT  = 1.0 / 50.0   # período de control (s) — 50 Hz
K   = 1.5           # ganancia proporcional
TOL = 0.005         # tolerancia de error de posición (m)

# --- Visualización en RViz ---
ENABLE_RVIZ_TRAJECTORY = True
MAX_EXECUTED_POINTS    = 5000

# --- Configuración articular inicial ---
Q0 = np.array([np.pi / 2, -1.5, -1.0, 1.2, 0.45, 0.0])


# ===========================================================================
# Utilidades de mapeo píxeles → robot
# ===========================================================================

def pixel_to_robot(px: float, py: float, img_size: int = IMAGE_SIZE) -> np.ndarray:
    """
    Convierte coordenadas en píxeles (imagen) al espacio cartesiano del robot.

    El eje X del robot corresponde al eje horizontal de la imagen.
    El eje Y del robot corresponde al eje vertical invertido (y_pixel crece
    hacia abajo; en el robot queremos que crezca hacia arriba).

    Retorna un vector [x, y, z] en metros.
    """
    nx = (px / img_size) - 0.5           # normalizado en [-0.5, 0.5]
    ny = -((py / img_size) - 0.5)        # invertir eje Y

    x = DRAW_CENTER[0] + nx * DRAW_WIDTH
    y = DRAW_CENTER[1] + ny * DRAW_HEIGHT
    z = DRAW_Z

    return np.array([x, y, z])


def build_xd(position: np.ndarray, quat: np.ndarray = None) -> np.ndarray:
    """
    Construye el vector de pose deseada [x, y, z, qw, qx, qy, qz].

    Si no se da cuaternión se usa orientación neutra (EE apuntando hacia abajo).
    """
    if quat is None:
        quat = np.array([0.0, 1.0, 0.0, 0.0])
    return np.concatenate([position, quat])


# ===========================================================================
# Utilidades de visualización RViz
# ===========================================================================


def xyz_to_point(p: np.ndarray) -> Point:
    """
    Convierte un vector [x, y, z] de NumPy a geometry_msgs/Point.
    """
    point = Point()
    point.x = float(p[0])
    point.y = float(p[1])
    point.z = float(p[2])
    return point


def create_line_strip_marker(
    frame: str,
    ns: str,
    marker_id: int,
    color=(0.0, 1.0, 0.0, 1.0),
    width: float = 0.006,
) -> Marker:
    """
    Crea un Marker tipo LINE_STRIP para visualizar trayectorias en RViz.

    Parámetros:
        frame: frame de referencia, por ejemplo "base_link".
        ns: namespace del marker.
        marker_id: ID único dentro del namespace.
        color: tupla RGBA.
        width: grosor de la línea en metros.
    """
    marker = Marker()
    marker.header.frame_id = frame
    marker.ns = ns
    marker.id = marker_id
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD

    marker.pose.orientation.w = 1.0
    marker.scale.x = float(width)

    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])

    marker.points = []
    return marker

# ===========================================================================
# Nodo ROS2
# ===========================================================================

class LetterDrawingNode(Node):

    def __init__(self):
        super().__init__("letter_drawing_node")

        # ===== Parámetros =====
        self.dt  = DT
        self.K   = K
        self.tol = TOL

        # ===== Publishers =====
        self.pub        = self.create_publisher(JointState, "joint_states", 10)
        self.marker_pub = self.create_publisher(Marker, "ee_marker", 10)

        # ===== Publishers para visualizar trayectoria en RViz =====
        self.desired_traj_pub  = self.create_publisher(Marker, "desired_trajectory_marker", 10)
        self.executed_traj_pub = self.create_publisher(Marker, "executed_trajectory_marker", 10)

        # ===== Nombres de joints =====
        self.jnames = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]

        # ===== Estado articular =====
        self.q = Q0.copy()

        # ===== JointState =====
        self.jstate      = JointState()
        self.jstate.name = self.jnames

        # ===== Marker EE =====
        self.marker = create_sphere_marker(
            frame     = "base_link",
            ns        = "ee",
            marker_id = 0,
            scale     = 0.03,
            color     = (1.0, 0.2, 0.0, 1.0),
        )

        # ===== Generar trayectorias =====
        self.trajectories_3d = self._generate_trajectories()
        self.get_logger().info(
            f"Trayectorias generadas: {len(self.trajectories_3d)}"
        )

        # ===== Markers de trayectoria para RViz =====
        self.desired_traj_marker = create_line_strip_marker(
            frame     = "base_link",
            ns        = "desired_trajectory",
            marker_id = 1,
            color     = (0.0, 0.4, 1.0, 0.8),
            width     = 0.004,
        )

        self.executed_traj_marker = create_line_strip_marker(
            frame     = "base_link",
            ns        = "executed_trajectory",
            marker_id = 2,
            color     = (1.0, 0.1, 0.1, 1.0),
            width     = 0.006,
        )

        self._executed_points = []

        if ENABLE_RVIZ_TRAJECTORY:
            self._publish_desired_trajectory_marker()

        # ===== Estado de ejecución =====
        self._traj_idx  = 0
        self._point_idx = 0
        self._phase     = "draw"      # "draw" | "liftoff" | "approach" | "done"

        self._current_xd = self._get_first_target()

        # ===== Timer de control =====
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("LetterDrawingNode V8 iniciado — comenzando dibujo")

    # -----------------------------------------------------------------------
    # Generación de trayectorias (pipeline V8)
    # -----------------------------------------------------------------------

    def _generate_trajectories(self) -> list:
        """
        Ejecuta el pipeline completo V8 y convierte las trayectorias
        al espacio cartesiano 3D del robot.

        Retorna una lista de arrays (N, 3) en metros, ordenados
        en el orden natural de escritura (izquierda a derecha).
        """
        font_name, font_path = FONTS[FONT_KEY]
        self.get_logger().info(f"Texto: '{LETTER}' | Fuente: {font_name}")

        # 1. Imagen
        file_path = plot_letter_pil(LETTER, font_path)

        # 2. Esqueleto
        img_skeleton, img_binary, skeleton, gray = process_image(file_path)
        if skeleton is None:
            self.get_logger().error("Error al procesar la imagen — sin esqueleto")
            return []

        # 3. Análisis del esqueleto
        endpoints, junctions, _ = analyze_skeleton_points(skeleton)
        self.get_logger().info(
            f"Extremos: {len(endpoints)} | Bifurcaciones: {len(junctions)}"
        )

        # 4. Segmentos ordenados (grafo del esqueleto)
        segments = skeleton_to_ordered_segments(
            skeleton,
            min_length    = 3,
            junction_dilate = 2,
            keep_cycles   = True,
        )
        self.get_logger().info(f"Segmentos originales: {len(segments)}")

        # 5. Fusión con lógica de escritura humana (V8)
        merged = merge_segments_human_writing_order(
            segments,
            max_endpoint_distance  = MERGE_MAX_DIST,
            max_angle              = MERGE_MAX_ANGLE,
            bridge_points          = MERGE_BRIDGE_PTS,
            min_branch_length      = 0,
            keep_closed_cycles     = True,
            open_closed_cycles     = True,
            cycle_connection_distance = MERGE_CYCLE_DIST,
            protect_short_isolated = True,
            expected_letter_gap    = MERGE_EXPECTED_LETTER_GAP,
            verbose                = True,
        )
        self.get_logger().info(f"Segmentos fusionados: {len(merged)}")

        # 6. Splines
        trajectories_px = segments_to_spline_trajectories(
            merged,
            smoothing  = SPLINE_SMOOTHING,
            num_points = SPLINE_NUM_PTS,
        )
        self.get_logger().info(f"Trayectorias spline: {len(trajectories_px)}")

        # 7. Convertir píxeles → metros
        trajectories_3d = []
        for traj_px in trajectories_px:
            traj_3d = np.array([
                pixel_to_robot(px, py) for px, py in traj_px
            ])
            trajectories_3d.append(traj_3d)

        return trajectories_3d

    # -----------------------------------------------------------------------
    # Objetivo inicial
    # -----------------------------------------------------------------------

    def _get_first_target(self) -> np.ndarray:
        if not self.trajectories_3d:
            return build_xd(DRAW_CENTER)

        first_point = self.trajectories_3d[0][0].copy()
        first_point[2] = LIFTOFF_Z
        return build_xd(first_point)

    # -----------------------------------------------------------------------
    # Visualización de trayectorias en RViz
    # -----------------------------------------------------------------------

    def _publish_desired_trajectory_marker(self):
        """
        Publica solamente la trayectoria deseada de escritura en el plano XY.

        No conecta los trazos con movimientos de subida/bajada.
        Cada trazo se publica como un Marker.LINE_STRIP independiente.
        """
        if not ENABLE_RVIZ_TRAJECTORY:
            return

        for i, traj in enumerate(self.trajectories_3d):
            if len(traj) == 0:
                continue

            marker = create_line_strip_marker(
                frame     = "base_link",
                ns        = "desired_trajectory",
                marker_id = i + 10,
                color     = (0.0, 0.4, 1.0, 0.8),
                width     = 0.004,
            )

            marker.header.stamp = self.get_clock().now().to_msg()

            # Solo puntos del trazo, todos proyectados al plano de escritura
            marker.points = []
            for p in traj:
                p_xy = p.copy()
                p_xy[2] = DRAW_Z
                marker.points.append(xyz_to_point(p_xy))

            self.desired_traj_pub.publish(marker)

    def _update_executed_trajectory_marker(self, x_real: np.ndarray):
        """
        Actualiza la trayectoria ejecutada, pero solo cuando el robot está
        escribiendo en el plano XY.

        No guarda puntos durante liftoff ni approach.
        """
        if not ENABLE_RVIZ_TRAJECTORY:
            return

        # Solo guardar puntos cuando realmente está dibujando
        if self._phase != "draw":
            return

        # Guardar solo x, y y proyectar z al plano de escritura
        p = np.array([x_real[0], x_real[1], DRAW_Z])

        self._executed_points.append(p)

        if len(self._executed_points) > MAX_EXECUTED_POINTS:
            self._executed_points.pop(0)

        self.executed_traj_marker.header.stamp = self.get_clock().now().to_msg()
        self.executed_traj_marker.points = [
            xyz_to_point(point) for point in self._executed_points
        ]

        self.executed_traj_pub.publish(self.executed_traj_marker)

    # -----------------------------------------------------------------------
    # FK
    # -----------------------------------------------------------------------

    def compute_fk(self, q: np.ndarray):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x

    # -----------------------------------------------------------------------
    # Loop de control (50 Hz)
    # -----------------------------------------------------------------------

    def update(self):
        if self._phase == "done":
            return

        xd = self._current_xd

        # Control cinemático diferencial
        dq     = compute_dq(self.q, xd, self.K)
        self.q = self.q + dq * self.dt

        # Publicar joints
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position     = self.q.tolist()
        self.pub.publish(self.jstate)

        # FK y error
        _, x  = self.compute_fk(self.q)
        error = np.linalg.norm(pose_error(xd, x))

        # Actualizar trayectoria ejecutada en RViz
        self._update_executed_trajectory_marker(x)

        # Publicar marker
        marker = set_marker_pose(self.marker, x, self)
        if marker is not None:
            self.marker = marker
            self.marker_pub.publish(self.marker)

        # Máquina de estados
        self._advance_state(error)

    # -----------------------------------------------------------------------
    # Máquina de estados de ejecución
    # -----------------------------------------------------------------------

    def _advance_state(self, error: float):
        """
        Avanza al siguiente punto / fase cuando el error es suficientemente pequeño.

        Estados:
            "draw"     → seguir puntos de la trayectoria actual a Z de contacto.
            "liftoff"  → levantar el EE al final de un trazo (pen-up).
            "approach" → volar al inicio del siguiente trazo (en altura de vuelo).
            "done"     → todas las trayectorias ejecutadas.
        """
        if error > self.tol:
            return

        if self._phase == "draw":
            self._point_idx += 1

            if self._point_idx < len(self.trajectories_3d[self._traj_idx]):
                # Siguiente punto dentro del trazo actual
                pt = self.trajectories_3d[self._traj_idx][self._point_idx]
                self._current_xd = build_xd(pt)
            else:
                # Trazo completado → levantar
                self._phase = "liftoff"
                liftoff     = self.trajectories_3d[self._traj_idx][-1].copy()
                liftoff[2]  = LIFTOFF_Z
                self._current_xd = build_xd(liftoff)
                self.get_logger().info(
                    f"Trazo {self._traj_idx + 1}/{len(self.trajectories_3d)} completado"
                )

        elif self._phase == "liftoff":
            self._traj_idx += 1

            if self._traj_idx < len(self.trajectories_3d):
                # Volar al inicio del siguiente trazo (manteniendo LIFTOFF_Z)
                next_start      = self.trajectories_3d[self._traj_idx][0].copy()
                next_start[2]   = LIFTOFF_Z
                self._current_xd = build_xd(next_start)
                self._phase      = "approach"
            else:
                self._phase = "done"
                self.get_logger().info("¡Dibujo completado!")

        elif self._phase == "approach":
            # Bajar al punto de contacto del nuevo trazo
            self._point_idx  = 0
            pt               = self.trajectories_3d[self._traj_idx][0]
            self._current_xd = build_xd(pt)
            self._phase      = "draw"


# ===========================================================================
# Entry point
# ===========================================================================

def main(args=None):
    rclpy.init(args=args)
    node = LetterDrawingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()