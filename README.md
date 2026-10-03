# UR5 ROS 2 Algorithms

Paquete ROS 2 (Python) con algoritmos de cinemática y control para el brazo
robótico **Universal Robots UR5**, visualizados en RViz2 o ejecutados en Gazebo:
cinemática directa e inversa, control cinemático diferencial con el Jacobiano,
control por optimización cuadrática (QP), escritura de palabras con el efector
final y trayectorias generadas a partir de instrucciones en lenguaje natural con
un LLM local.

## Estructura

```text
ur5_algoritmos/
├── launch/ur5_view.launch.py        # UR5 en RViz2 (robot_state_publisher + rviz2)
└── ur5_algoritmos/
    ├── funciones/                   # Librerías (sin nodos)
    │   ├── fk_functions.py               cinemática directa (DH) y utilidades de pose
    │   ├── ik_functions.py               cinemática inversa
    │   ├── kine_control_functions.py     Jacobiano, error de pose, ley de control
    │   ├── QP_functions.py               control por QP (OSQP)
    │   ├── letter_trajectory_functions.py  texto → esqueleto → trayectorias suaves
    │   ├── p_llm_interface.py            instrucción → figura 3D con un LLM (Ollama)
    │   └── markers.py                    marcadores de RViz2
    ├── fk_ur5.py / fk_ur5_gazebo.py      # Nodos ejecutables
    ├── ik_ur5.py
    ├── kine_control_ur5.py
    ├── QP_ur5.py
    ├── letter_trajectory_main.py
    └── p_test_llm_track.py / p_test_llm_track_gazebo.py
```

## Nodos

| Ejecutable | Qué hace |
|---|---|
| `fk_ur5` | Cinemática directa: publica `joint_states` y la pose del efector como marcador |
| `ik_ur5` | Cinemática inversa hacia una pose objetivo |
| `kine_control_ur5` | Control cinemático diferencial (Jacobiano) a 50 Hz |
| `QP_ur5` | Control cinemático resuelto como QP (OSQP), con límites de posición y velocidad articular |
| `letter_trajectory` | Escribe una palabra con el efector: esqueletiza el texto, ordena los trazos como escritura humana y los sigue con levantamientos de "lápiz" entre trazos |
| `p_test_llm_track` | Pide una figura en lenguaje natural (p. ej. "un círculo de 5 cm"), un LLM la convierte en parámetros y el robot la sigue |
| `*_gazebo` | Variantes que leen `/joint_states` y mandan comandos a `/joint_trajectory_controller/joint_trajectory` en Gazebo |

Los nodos para RViz2 publican en `joint_states` y en marcadores (`ee_marker`,
trayectoria deseada y ejecutada) para comparar lo planeado con lo ejecutado.

## Requisitos

- Ubuntu 22.04 + **ROS 2 Humble**
- `ur_description` (paquete `ros-humble-ur-description`)
- `pip install numpy scipy osqp opencv-python pillow scikit-image requests`
- Para los nodos con LLM: [Ollama](https://ollama.com) corriendo en local con el modelo `mistral`

## Compilar y ejecutar

```bash
mkdir -p ~/ur5_ws/src && cd ~/ur5_ws/src
git clone https://github.com/Migueltoh2309/UR5_ROS2_Algorithms.git
cd ~/ur5_ws
colcon build --symlink-install
source install/setup.bash

# Terminal 1: UR5 en RViz2
ros2 launch ur5_algoritmos ur5_view.launch.py

# Terminal 2: un nodo, por ejemplo
ros2 run ur5_algoritmos kine_control_ur5
ros2 run ur5_algoritmos letter_trajectory
```

## Autor

MiTo Olórtegui Huamán — UTEC
