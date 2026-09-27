"""ROS 2 узел резервной одометрии трамвая (без GNSS и IMU в основном контуре).

Подписки:   /vehicle/front_bogie_velocity, /vehicle/rear_bogie_velocity  (VelocitySensor, км/ч)
            /vehicle/driver_position_cmd                                (DriverControllerCommand)
            /sensing/gnss/{master,rover}/fix  (NavSatFix) - только окно выставки,
            сразу после него подписки удаляются.
Публикации: /result/velocity    (tram_vehicle_msgs/VelocitySensor, м/с)
            /result/position    (nav_msgs/Odometry, base_link в MGRS 37U local, м)
            /result/diagnostics (diagnostic_msgs/DiagnosticArray, 1 Гц: юз, коррекции, задержка)
Штамп каждого выхода равен header.stamp входного сообщения, которое его породило.
"""
import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import VelocitySensor

try:  # в пакете сообщений чекера этого типа может не быть
    from tram_vehicle_msgs.msg import DriverControllerCommand
except ImportError:
    DriverControllerCommand = None

from .estimator import DEFAULT_PARAMS, Estimator
from .route import DATA_DIR, load_queues, load_routes, load_stops, load_traction, load_vehicles


def stamp_sec(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class TramOdometryNode(Node):
    def __init__(self):
        super().__init__('tram_odometry')
        self.declare_parameter('data_dir', '')
        self.declare_parameter('vehicle_id', '', ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('initial_x', float('nan'))
        self.declare_parameter('initial_y', float('nan'))
        self.declare_parameter('initial_yaw', float('nan'))
        self.declare_parameter('gnss_wait', 10.0)
        for k, v in DEFAULT_PARAMS.items():
            self.declare_parameter(k, v)
        gp = self.get_parameter
        data_dir = gp('data_dir').value or DATA_DIR
        params = {k: gp(k).value for k in DEFAULT_PARAMS}
        vehicles = load_vehicles(data_dir + '/vehicles.json')
        vid = str(gp('vehicle_id').value)
        if vid and vid in vehicles and params['wheel_scale'] == DEFAULT_PARAMS['wheel_scale']:
            params['wheel_scale'] = vehicles[vid]
        model = (load_routes(data_dir + '/routes.json'), params, load_stops(data_dir + '/stops.json'),
                 load_traction(data_dir + '/traction.json'), load_queues(data_dir + '/stops.json'))
        self.new_estimator = lambda: Estimator(*model)
        self.est = self.new_estimator()
        self.last_in = None      # самый поздний штамп входа (для распознавания нового bag)
        self.aligned_logged = False
        self.map_frame = gp('map_frame').value
        self.base_frame = gp('base_frame').value
        self.init_xy = (gp('initial_x').value, gp('initial_y').value, gp('initial_yaw').value)
        self.gnss_wait = gp('gnss_wait').value
        self.t_first = None
        self.get_logger().info('data=%s vehicle=%s wheel_scale=%.5f routes=%d' % (
            data_dir, vid or '-', params['wheel_scale'], len(self.est.routes)))

        sub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=100)
        self.pub_v = self.create_publisher(VelocitySensor, '/result/velocity', 10)
        self.pub_p = self.create_publisher(Odometry, '/result/position', 10)
        self.pub_d = self.create_publisher(DiagnosticArray, '/result/diagnostics', 10)
        self.create_subscription(VelocitySensor, '/vehicle/front_bogie_velocity',
                                 lambda m: self.on_wheel(m, 'front'), sub_qos)
        self.create_subscription(VelocitySensor, '/vehicle/rear_bogie_velocity',
                                 lambda m: self.on_wheel(m, 'rear'), sub_qos)
        if DriverControllerCommand is not None:
            self.create_subscription(DriverControllerCommand, '/vehicle/driver_position_cmd', self.on_cmd, sub_qos)
        else:
            self.get_logger().warn('tram_vehicle_msgs has no DriverControllerCommand: running on wheel '
                                   'speeds only (traction model disabled for the notch input)')
        self.sub_qos = sub_qos
        self.gnss_subs = []
        self.open_gnss()
        self.lat = []            # задержки обработки, с (начало callback -> публикация)
        self.n_out = 0
        self.last_out_t = None
        self.create_timer(1.0, self.publish_diag)

    def open_gnss(self):
        self.gnss_subs = [
            self.create_subscription(NavSatFix, '/sensing/gnss/master/fix',
                                     lambda m: self.on_fix(m, 'master'), self.sub_qos),
            self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix',
                                     lambda m: self.on_fix(m, 'rover'), self.sub_qos),
        ]

    def check_new_bag(self, t):
        """Скачок штампа назад больше 5 с или вперёд больше 30 с - начался новый bag (узел не
        перезапускали): оценщик сбрасывается, подписки на GNSS открываются заново для выставки."""
        if self.last_in is not None and (t < self.last_in - 5.0 or t > self.last_in + 30.0):
            self.get_logger().warn('input time jumped %.1f s: new bag, estimator reset' % (t - self.last_in))
            self.est = self.new_estimator()
            self.t_first = None
            self.aligned_logged = False
            self.last_in = t
            if not self.gnss_subs:
                self.open_gnss()
            return
        self.last_in = t if self.last_in is None else max(self.last_in, t)

    # ------------------------------------------------------------------ callbacks
    def on_wheel(self, msg, bogie):
        c0 = time.perf_counter()
        t = stamp_sec(msg)
        self.check_new_bag(t)
        self.est.on_wheel(t, bogie, float(msg.velocity))
        self.after_input(t, msg.header.stamp, c0)

    def on_cmd(self, msg):
        c0 = time.perf_counter()
        t = stamp_sec(msg)
        self.check_new_bag(t)
        self.est.on_cmd(t, int(msg.position))
        self.after_input(t, msg.header.stamp, c0)

    def on_fix(self, msg, antenna):
        if not self.gnss_subs:
            return
        self.check_new_bag(stamp_sec(msg))
        self.est.on_gnss(stamp_sec(msg), antenna, float(msg.latitude), float(msg.longitude),
                         int(msg.status.status))
        self.alignment_done()

    def alignment_done(self):
        """После выставки: без gnss_correction подписки на GNSS закрываются, с ней остаются -
        редкие фиксы по ходу рейса мягко поправляют положение."""
        if self.aligned_logged or not self.est.initialized or self.est.aligning:
            return
        self.aligned_logged = True
        where = (self.est.route.name if self.est.route else 'none', self.est.s)
        if self.est.p['gnss_correction']:
            self.get_logger().info('alignment done (%s, s=%.1f m); GNSS kept for gentle corrections' % where)
            return
        for sub in self.gnss_subs:
            self.destroy_subscription(sub)
        self.gnss_subs = []
        self.get_logger().info('alignment done (%s, s=%.1f m); GNSS subscriptions closed' % where)

    def after_input(self, t, stamp, c0):
        if self.t_first is None:
            self.t_first = t
        if not self.est.initialized and t - self.t_first > self.gnss_wait:
            x, y, yaw = self.init_xy
            if math.isfinite(x) and math.isfinite(y):
                self.est.init_from_xy(t, x, y, yaw if math.isfinite(yaw) else None)
                self.get_logger().warn('no GNSS alignment - using initial_x/initial_y')
        if self.gnss_subs and self.est.initialized and not self.est.aligning:
            self.alignment_done()
        st = self.est.state(t)
        if st is None:
            return
        v, x, y, z, yaw, s = st
        mv = VelocitySensor()
        mv.header.stamp = stamp
        mv.header.frame_id = self.base_frame
        mv.velocity = float(v)
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = self.map_frame
        od.child_frame_id = self.base_frame
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = x, y, z
        od.pose.pose.orientation.z = math.sin(yaw / 2)
        od.pose.pose.orientation.w = math.cos(yaw / 2)
        sa = self.est.sigma_s() ** 2          # дисперсия вдоль пути -> x/y по курсу
        sc = 0.3 ** 2                         # поперёк пути (точность карты)
        c, sn = math.cos(yaw), math.sin(yaw)
        cov = [0.0] * 36
        cov[0] = sa * c * c + sc * sn * sn
        cov[1] = cov[6] = (sa - sc) * c * sn
        cov[7] = sa * sn * sn + sc * c * c
        cov[14] = 0.3 ** 2
        cov[21] = cov[28] = 1e3
        cov[35] = 0.01
        od.pose.covariance = cov
        od.twist.twist.linear.x = float(v)
        tcov = [0.0] * 36
        tcov[0] = self.est.pv
        od.twist.covariance = tcov
        self.pub_v.publish(mv)
        self.pub_p.publish(od)
        self.lat.append(time.perf_counter() - c0)
        self.n_out += 1

    def publish_diag(self):
        d = self.est.diagnostics()
        lat = sorted(self.lat) or [0.0]
        st = DiagnosticStatus()
        st.name = 'tram_odometry'
        st.hardware_id = 'tram'
        st.level = DiagnosticStatus.WARN if d['slip'] or not d['initialized'] else DiagnosticStatus.OK
        st.message = 'slip/sensor anomaly' if d['slip'] else ('ok' if d['initialized'] else 'waiting for alignment')
        d.update(outputs_per_s=self.n_out, latency_mean_ms=1e3 * sum(lat) / len(lat),
                 latency_max_ms=1e3 * lat[-1], latency_p99_ms=1e3 * lat[int(0.99 * (len(lat) - 1))])
        st.values = [KeyValue(key=k, value=('%.4f' % v) if isinstance(v, float) else str(v)) for k, v in d.items()]
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [st]
        self.pub_d.publish(arr)
        if self.n_out:
            self.get_logger().info('out %d Hz, latency mean %.3f / max %.3f ms, s=%.1f sigma=%.1f m, '
                                   'corr=%d slip=%s' % (self.n_out, d['latency_mean_ms'], d['latency_max_ms'],
                                                         d['s'], d['sigma_s'], d['n_corrections'], d['slip']))
        self.lat, self.n_out = [], 0


def main(args=None):
    rclpy.init(args=args)
    node = TramOdometryNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # Ctrl-C может закрыть контекст во время spin (RCLError)
        if rclpy.ok():
            raise
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
