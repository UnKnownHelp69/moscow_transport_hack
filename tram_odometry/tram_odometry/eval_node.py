"""Монитор точности для жюри и демо, в оценке не участвует.

Слушает /result/velocity, /result/position и GNSS, сопоставляет по header.stamp (<= 0.05 с,
как чекер), каждые 10 с печатает метрики, при остановке - итог. Эталон положения - base_link
из двух антенн по tf организаторов:
    base = master + 9.873 * unit(rover - master), z = mean(alt) - 3.0   (MGRS 37U local)
Эталон скорости - |/sensing/gnss/master/vel linear.xy| (только при валидном фиксе).
Параметр csv_path: при остановке записать сопоставленные точки в CSV.
"""
import bisect
import math

import rclpy
from rclpy.executors import ExternalShutdownException
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from tram_vehicle_msgs.msg import VelocitySensor

from .geo import latlon_to_mgrs_local

TOL = 0.05


def ts(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class Series:
    def __init__(self):
        self.t, self.v = [], []

    def add(self, t, v):
        if self.t and t < self.t[-1]:
            i = bisect.bisect(self.t, t)
            self.t.insert(i, t)
            self.v.insert(i, v)
        else:
            self.t.append(t)
            self.v.append(v)

    def nearest(self, t):
        i = bisect.bisect(self.t, t)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(self.t) and (best is None or abs(self.t[j] - t) < abs(self.t[best] - t)):
                best = j
        if best is None or abs(self.t[best] - t) > TOL:
            return None
        return self.v[best]


class EvalNode(Node):
    def __init__(self):
        super().__init__('tram_odometry_eval')
        self.declare_parameter('csv_path', '')
        q = qos_profile_sensor_data
        self.out_v, self.out_p = Series(), Series()
        self.rover = Series()
        self.fix_t = []
        self.rows_p, self.rows_v = [], []
        self.pending_ref, self.pending_vel = [], []
        self.create_subscription(VelocitySensor, '/result/velocity', lambda m: self.out_v.add(ts(m), m.velocity), q)
        self.create_subscription(Odometry, '/result/position', self.on_pos, q)
        self.create_subscription(NavSatFix, '/sensing/gnss/master/fix', self.on_master, q)
        self.create_subscription(NavSatFix, '/sensing/gnss/rover/fix', self.on_rover, q)
        self.create_subscription(TwistStamped, '/sensing/gnss/master/vel', self.on_vel, q)
        self.n_out = 0
        self.create_timer(10.0, self.report)

    def on_pos(self, m):
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        self.out_p.add(ts(m), (p.x, p.y, p.z, 2 * math.atan2(q.z, q.w)))
        self.n_out += 1

    def on_rover(self, m):
        if m.status.status == 2 and m.latitude != 0.0:
            x, y = latlon_to_mgrs_local(m.latitude, m.longitude)
            self.rover.add(ts(m), (float(x), float(y), m.altitude))

    def on_master(self, m):
        if m.status.status != 2 or m.latitude == 0.0 or not math.isfinite(m.latitude):
            return
        t = ts(m)
        self.fix_t.append(t)
        x, y = latlon_to_mgrs_local(m.latitude, m.longitude)
        self.pending_ref.append((t, float(x), float(y), m.altitude))
        self.match()

    def on_vel(self, m):
        self.pending_vel.append((ts(m), math.hypot(m.twist.linear.x, m.twist.linear.y)))
        self.match()

    def match(self):
        # ждём ~0.5 с, чтобы пришли выходы с близкими штампами
        horizon = (self.out_p.t[-1] if self.out_p.t else 0.0) - 0.5
        keep = []
        for t, mx, my, alt in self.pending_ref:
            if t > horizon:
                keep.append((t, mx, my, alt))
                continue
            r = self.rover.nearest(t)
            o = self.out_p.nearest(t)
            if r is None or o is None:
                continue
            dx, dy = r[0] - mx, r[1] - my
            L = math.hypot(dx, dy)
            if abs(L - 12.436) > 0.5:
                continue
            bx, by, bz = mx + 9.873 * dx / L, my + 9.873 * dy / L, 0.5 * (alt + r[2]) - 3.0
            ex, ey, ez = o[0] - bx, o[1] - by, o[2] - bz
            at = ex * math.cos(o[3]) + ey * math.sin(o[3])
            ct = -ex * math.sin(o[3]) + ey * math.cos(o[3])
            self.rows_p.append((t, bx, by, bz, o[0], o[1], o[2], math.sqrt(ex * ex + ey * ey + ez * ez), at, ct))
        self.pending_ref = keep
        keep = []
        for t, v in self.pending_vel:
            if t > horizon:
                keep.append((t, v))
                continue
            i = bisect.bisect(self.fix_t, t)
            if not any(0 <= j < len(self.fix_t) and abs(self.fix_t[j] - t) < 0.2 for j in (i - 1, i)):
                continue
            o = self.out_v.nearest(t)
            if o is not None:
                self.rows_v.append((t, v, o, o - v))
        self.pending_vel = keep

    def summary(self):
        s = {}
        if self.rows_v:
            e = [r[3] for r in self.rows_v]
            s['v_rmse'] = math.sqrt(sum(x * x for x in e) / len(e))
            s['v_mae'] = sum(abs(x) for x in e) / len(e)
            s['v_bias'] = sum(e) / len(e)
        if self.rows_p:
            e = [r[7] for r in self.rows_p]
            at = [r[8] for r in self.rows_p]
            ct = [r[9] for r in self.rows_p]
            s['p_rmse'] = math.sqrt(sum(x * x for x in e) / len(e))
            s['p_mean'] = sum(e) / len(e)
            s['p_max'] = max(e)
            s['along_rmse'] = math.sqrt(sum(x * x for x in at) / len(at))
            s['along_max'] = max(abs(x) for x in at)
            s['cross_rmse'] = math.sqrt(sum(x * x for x in ct) / len(ct))
            s['n_pos'] = len(e)
        return s

    def report(self):
        s = self.summary()
        self.get_logger().info('result rate %.1f Hz | %s' % (
            self.n_out / 10.0, ' '.join('%s=%.3f' % (k, v) for k, v in s.items())))
        self.n_out = 0

    def finish(self):
        self.pending_ref = [p for p in self.pending_ref]
        s = self.summary()
        print('FINAL ' + ' '.join('%s=%.3f' % (k, v) for k, v in s.items()), flush=True)
        path = self.get_parameter('csv_path').value
        if path:
            with open(path, 'w') as f:
                f.write('t,ref_x,ref_y,ref_z,est_x,est_y,est_z,err3d,along,cross\n')
                for r in self.rows_p:
                    f.write(','.join('%.4f' % x for x in r) + '\n')
            print('wrote %s' % path, flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = EvalNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:  # Ctrl-C может закрыть контекст во время spin (RCLError)
        if rclpy.ok():
            raise
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
