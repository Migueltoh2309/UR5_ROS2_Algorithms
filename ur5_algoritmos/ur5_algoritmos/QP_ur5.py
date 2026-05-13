#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker

# Personal
from ur5_algoritmos.fk_functions import *
from ur5_algoritmos.ik_functions import *
from ur5_algoritmos.kine_control_functions import *
from ur5_algoritmos.QP_functions import *
from ur5_algoritmos.markers import *


class UR5ControlNode(Node):

    def __init__(self):
        super().__init__('ur5_kinecontrol_node')

        # ===== PARAMETROS =====
        self.dt = 1.0 / 50.0   # 50 Hz
        self.K = 1.5           # ganancia
        self.lamb = 0.01       # regularización QP

        # ===== Publisher joints =====
        self.pub = self.create_publisher(JointState, 'joint_states', 10)

        # ===== Marker =====
        self.marker_pub = self.create_publisher(Marker, 'ee_marker', 10)

        # ===== Joint names =====
        self.jnames = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        # ===== Estado inicial =====
        self.q = np.array([np.pi/2, -2.14, -1.34, -1.23, np.pi/2, 0.0])

        # ===== JointState =====
        self.jstate = JointState()
        self.jstate.name = self.jnames

        # ===== Marker =====
        self.marker = create_sphere_marker(
            frame="base_link",
            ns="ee",
            marker_id=0,
            scale=0.05,
            color=(0.0, 0.0, 1.0, 1.0)
        )

        # ===== Trayectoria circular =====
        self.t = 0.0
        self.radius = 0.1
        self.omega = 1.0  # rad/s
        self.center = np.array([0.4, 0.24, 0.5])

        # ===== Timer =====
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("UR5 Control Node (QP) iniciado")

    # =========================
    # FK
    # =========================
    def compute_fk(self, q):
        T = fkine_ur5(q)
        x = TF2xyzquat(T)
        return T, x

    # =========================
    # LOOP
    # =========================
    def update(self):

        # ===== 1. Trayectoria =====
        xd = circular_trajectory(self.t, self.center, self.radius, self.omega)

        # ===== 2. CONTROL QP =====
        dq = compute_dq_qp(
            fkine=fkine_ur5,
            jacobian_func=numerical_jacobian,
            TF2xyzquat=TF2xyzquat,
            q=self.q,
            xd=xd,
            K=self.K,
            lamb=self.lamb,
            dt=self.dt
            )

        # ===== 3. Integración =====
        self.q = self.q + dq * self.dt

        # ===== 4. Publicar joints =====
        self.jstate.header.stamp = self.get_clock().now().to_msg()
        self.jstate.position = self.q.tolist()
        self.pub.publish(self.jstate)

        # ===== 5. FK =====
        T, x = self.compute_fk(self.q)

        # ===== 6. Marker =====
        marker = set_marker_pose(self.marker, x, self)

        if marker is not None:
            self.marker = marker
            self.marker_pub.publish(self.marker)

        # ===== 7. Tiempo =====
        self.t += self.dt

        # ===== Debug =====
        error = np.linalg.norm(pose_error(xd, x))
        print(f"Error: {error:.4f}")


def main(args=None):
    rclpy.init(args=args)

    node = UR5ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()