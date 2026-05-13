from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ur5_algoritmos'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),

    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),

        (os.path.join('share', package_name, 'urdf'),
            glob('urdf/*.xacro')),
    ],

    install_requires=['setuptools'],
    zip_safe=True,

    maintainer='utec',
    maintainer_email='molortegui@utec.edu.pe',
    description='UR5 control with custom algorithms',
    license='Apache License 2.0',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    entry_points={
        'console_scripts': [
            'fk_ur5 = ur5_algoritmos.fk_ur5:main',
            'fk_ur5_gazebo = ur5_algoritmos.fk_ur5_gazebo:main',
            'ik_ur5 = ur5_algoritmos.ik_ur5:main',
            'kine_control_ur5 = ur5_algoritmos.kine_control_ur5:main',
            'QP_ur5 = ur5_algoritmos.QP_ur5:main',
            'p_test_llm_track = ur5_algoritmos.p_test_llm_track:main',
            'p_test_llm_track_gazebo = ur5_algoritmos.p_test_llm_track_gazebo:main',
        ],
    },
)
