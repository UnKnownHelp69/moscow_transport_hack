from glob import glob

from setuptools import setup

package_name = 'tram_odometry'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    package_data={package_name: ['data/*.json', 'data/*.md']},
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml') + glob('config/*.json')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mos_trap team',
    maintainer_email='mgareev111@gmail.com',
    description='Резервная одометрия трамвая без GNSS и IMU',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'tram_odometry_node = tram_odometry.node:main',
            'tram_odometry_eval = tram_odometry.eval_node:main',
        ],
    },
)
