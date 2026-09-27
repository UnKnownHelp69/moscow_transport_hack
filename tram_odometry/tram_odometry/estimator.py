"""Оценщик резервной одометрии: без ROS, причинный, по событиям.

Состояние: вариант маршрута, длина дуги s вдоль него, продольная скорость v.
Входы: скорости тележек (км/ч), позиция контроллера, фиксы GNSS (только окно выставки).
Выход: v и (x, y, z) base_link в MGRS 37U local.
"""
import math

import numpy as np

from .geo import latlon_to_mgrs_local

KMH = 1.0 / 3.6
MASTER_X, ROVER_X = -9.873, 2.563      # x антенн в base_link, м
BASELINE = ROVER_X - MASTER_X            # 12.436 м

DEFAULT_PARAMS = {
    'input_lag_tol': 0.5,      # с: допустимое опоздание входа относительно самого нового (топики перемежаются)
    'init_window': 3.0,        # с: окно GNSS для выставки после первого валидного фикса
    'wheel_scale': 1.0,        # истинная скорость = scale * скорость колеса
    'wheel_timeout': 0.5,      # с: более старое показание тележки не используется
    # --- фильтр Калмана скорости: прогноз по модели тяги + обновления по тележкам
    'use_model': True,
    'accel_std': 0.3,          # (м/с)/sqrt(с): шум процесса, P += accel_std^2 * dt
    'wheel_std': 0.05,         # м/с: шум скорости колеса
    'distance_innovation_gain': 0.5, # поправка пути трапецией при каждом принятом обновлении скорости
    'distance_innovation_max_dt': 0.2, # с: предел интервала от предыдущего принятого обновления
    'gate_min': 0.7,           # м/с: минимальный допуск показания одиночной тележки
    'gate_sigma': 4.0,
    'recover_time': 2.0,       # с: сколько модель может перекрывать одиночную тележку или режим юза
    'agree_tol': 0.5,          # м/с: тележки, согласные в этих пределах, принимаются вместе
    'slip_accel_max': 6.0,     # м/с^2: экстренное торможение проходит, большие скачки отвергаются
    'spin_margin': 1.5,        # м/с^2: при тяге ускорение колеса > модель + это = буксование
    'slip_window': 0.3,        # с: окно оценки ускорения колеса
    'slip_exit_tol': 0.3,      # м/с: выход из юза, когда колёса снова в пределах
    'slip_exit_rate': 0.5,     #      tol + rate * время_в_юзе от прогноза модели
    'init_max_xt': 8.0,        # м: макс. поперечное отклонение маршрута при выставке
    'init_fallback_xt': 40.0,  # м: если ближе init_max_xt ничего нет, привязка в пределах этого
    # --- контрольные точки: остановки у станций (повторяются с точностью ~0.2 м)
    'stops_enabled': True,
    'stop_speed': 0.3 / 3.6,   # м/с: обе тележки ниже - стоянка
    'stop_dwell': 5.0,         # с стоянки, после которых остановка используется как контрольная точка
    'stop_min_travel': 0.0,    # м после выставки до первой станции
    'stop_repeat_dist': 50.0,  # м: одна и та же станция не используется повторно раньше этого пути
    'anchor_scale': True,      # масштаб пути регрессией s станций по одометру
    'anchor_scale_std': 0.005, # априорное СКО масштаба (ridge)
    'anchor_stop_var': 0.25,   # м^2: дисперсия станции как опорной точки
    'anchor_use_init': True,   # точка выставки - первая опорная точка
    'anchor_init_var': 25.0,   # м^2: её дисперсия
    'anchor_deadband': 0.002,  # изменения масштаба меньше 0.2 % игнорируются
    'stop_min_bags': 3,        # станция должна встречаться минимум в стольких train-прогонах
    'stop_max_mad': 0.5,       # м: и повторяться с такой точностью
    'stop_meas_std': 1.0,      # м: СКО положения станции
    'stop_gate_min': 5.0,      # м: минимальный допуск ассоциации
    'stop_gate_sigma': 3.0,    # допуск = max(min, k * sigma_s)
    'stop_gate_max': 1e9,      # м: максимальный допуск ассоциации
    'stop_bootstrap_gate': 15.0, # м: допуск до первой станции после сомнительной выставки; 0 = выкл.
    'branch_enabled': True,    # развилка на конечной (кольцо или тупик) по скорости колёс
    'branch_speed': 5.0,       # м/с: быстрее этого после развилки - тупик (в кольце <= 4.6 м/с)
    'branch_start': 20.0,      # м после конца карты: начало окна решения
    'branch_window': 200.0,    # м после конца карты: конец окна решения
    # --- GNSS после выставки (если фиксы приходят): мягкая поправка пути и выбор ветки
    'gnss_correction': True,   # False - GNSS только для выставки
    'gnss_corr_std': 3.0,      # м: считаемая погрешность фикса вдоль пути (с запасом на шум)
    'gnss_corr_max_xt': 2.5,   # м: фикс дальше от пути отбрасывается
    'gnss_corr_max_xt_tail': 5.0, # м: то же за развилкой западной конечной (соседние пути тупиков)
    'gnss_corr_gate': 15.0,    # м: мин. допуск невязки; допуск = max(этого, 3 * sqrt(sigma_s^2 + std^2))
    'gnss_corr_min_dt': 1.0,   # с: не чаще одной поправки за это время
    'gnss_corr_window': 60.0,  # м: поиск проекции вокруг прогноза
    'gnss_corr_pair_dt': 0.5,  # с: окно сверки фиксов master и rover
    'gnss_corr_pair_tol': 0.5, # м: допуск к базе 12.436 м (плюс v * dt между фиксами)
    'gnss_corr_holdoff': 10.0, # с: после расхождения антенн фиксы не используются
    'queue_mode': 'soft',      # места очередей и светофоров: 'off' | 'skip' | 'soft'
    'queue_prior': 1.0,        # априорный вес места очереди относительно станции
    'queue_min_std': 2.0,      # м: минимальное СКО места очереди
    'recover_agree': 3.0,      # м: восстановление: две подряд остановки вне допуска с одинаковым
    'recover_window': 2000.0,  #    сдвигом (в пределах этого, на пути <= этого) - сдвиг принимается
    'recover_max': 0.0,        # м: макс. скачок восстановления; 0 = выкл. (ложные восстановления у очередей)
    'drift_rate': 0.003,       # sigma_s растёт на эту долю пройденного пути
    'init_std': 1.0,           # м: sigma_s после выставки
    # --- фильтр пути: 'pf' - фильтр частиц по (s, масштаб колёс), 'kf' - одна гипотеза
    'along_mode': 'kf',
    'pf_n': 400,
    'pf_scale_std': 0.002,     # априорный разброс масштаба частиц
    'pf_scale_jitter': 0.0005, # шум масштаба после ресэмплинга
    'pf_s_jitter': 0.3,        # м: шум s после ресэмплинга
    'pf_p_station': 0.7,       # вероятность, что остановка (>= dwell) у известной станции
    'pf_off_density': 1.0 / 300.0,  # 1/м: плотность гипотезы 'остановился в другом месте'
    'pf_seed': 0,
    'pf_explore': 0.0,         # доля частиц, разбрасываемых вокруг среднего на каждой остановке
    'pf_explore_min': 20.0,    # м: полуширина разброса (или 3 sigma, если больше)
    'init_fallback_std': 30.0, # м: начальная sigma_s после привязки издалека
    'init_edge': 15.0,         # м: выставка ближе этого к началу маршрута - сомнительная
    'init_fallback_tie': 10.0, # м: при сомнительной выставке маршруты ближе этого считаются равными
    'init_prefix_candidates': True, # все локальные проекции на самопересекающиеся маршруты
    'init_end_guard': True,    # маршрут, который продолжается, лучше того, что здесь заканчивается
    'init_observed_priority': True, # маршруты из train-прогонов важнее достроенных только по OSM
    # --- адаптация масштаба по невязкам станций (выкл., заменена anchor_scale)
    'scale_adapt': False,
    'scale_gain': 0.5,
    'scale_min_dist': 300.0,   # м пути между обновлениями масштаба
    'scale_limit': 0.02,
}


class Estimator:
    def __init__(self, routes, params=None, stops=None, traction=None, queues=None):
        self.routes = routes
        self.traction = traction
        self.p = dict(DEFAULT_PARAMS)
        self.p.update(params or {})
        self.route = None
        self.s = 0.0
        self.v = 0.0
        self.t = None                 # время последнего прогноза
        self.last_update_t = None     # время предыдущего принятого обновления скорости (узел трапеции)
        self.t_start = None           # первое время оценщика (узел до первого обновления)
        self.wheel = {}               # тележка -> (t, v м/с)
        self.notch = 0
        self.gnss_t0 = None
        self.last_fix = {}            # антенна -> (t, x, y)
        self.last_in = {}             # поток -> время последнего принятого входа
        self.initialized = False
        self.aligning = True
        self.odo = 0.0                # интеграл скорости (м) с поправками, не зависит от маршрута
        self.align_odo = 0.0          # одометр в момент выставки
        self.fix_hist = []            # (odo, x, y, смещение антенны) во время выставки
        self.scale = 1.0              # масштаб пути
        self.var0 = self.p['init_std'] ** 2
        self.d_since = 0.0            # путь после последней коррекции
        self.stand_t0 = None
        self.stand_used = False
        self.n_corrections = 0
        self.last_innovation = 0.0
        self.last_station = None     # (s, odo) последней принятой станции
        self.scale_anchors = []       # (odo, s, var) опорные точки регрессии масштаба
        self.pending = None           # (odo, невязка, cp) последней остановки вне допуска
        self.ps = None                # частицы: длина дуги
        self.pk = None                # частицы: масштаб колёс
        self.pw = None                # веса частиц
        self.rng = np.random.default_rng(self.p['pf_seed'])
        self.n_recoveries = 0
        self.pv = 1.0                 # дисперсия скорости
        self.cmd_hist = []            # (t, позиция) для задержки модели тяги
        self.reject_since = {}        # тележка -> время начала отвергаемых показаний
        self.n_rejected = 0
        self.slip = False             # диагностика: тележки расходятся / показание отвергнуто
        self.slip_mode = False        # колёсам не верим, скорость ведёт модель тяги
        self.slip_t0 = None
        self.n_slip = 0
        self.whist = {}               # тележка -> список (t, z) для оценки ускорения
        self.stop_s = {}              # маршрут -> отсортированные s станций
        self.terminal_s = {}          # места стоянки на конечных (вне карты); для масштаба не используются
        self.queue_s = {}             # маршрут -> [(s, СКО)] мест очередей и светофоров
        self.n_queue = 0
        self.n_branch = 0
        self.n_gnss_corr = 0          # принятые поправки по GNSS после выставки
        self.gnss_corr_t = None       # время последней такой поправки
        self.gnss_branch_votes = 0    # подряд идущие фиксы за другую ветку
        self.gnss_last = {}           # антенна -> (t, x, y) последнего RTK-фикса
        self.gnss_bad_t = None        # время последнего расхождения антенн
        self.gnss_undo = None         # (t фикса, сдвиг s, var0, d_since) последней поправки
        self.branch_by_gnss = False   # ветку выбрал GNSS - правило по скорости не трогает
        if stops and self.p['stops_enabled']:
            self._index_stops(stops)
        if queues and self.p['queue_mode'] != 'off':
            for r in self.routes:
                qs = []
                for q in queues:
                    s, xt, _ = r.project(q['x'], q['y'])
                    if xt < 1.5:
                        qs.append((s, max(q['std'], self.p['queue_min_std'])))
                self.queue_s[r.name] = qs

    # ------------------------------------------------------------------ входы
    def _valid_time(self, t, stream=None):
        """Время конечное, монотонно внутри своего потока и не старше самого нового входа более
        чем на `input_lag_tol`. Штампы топиков в bag перемежаются на 0.1-0.3 с, общая проверка
        монотонности отбросила бы большую часть показаний колёс."""
        if not (isinstance(t, (int, float)) and math.isfinite(t)):
            return False
        if self.t is not None and t < self.t - self.p['input_lag_tol']:
            return False
        if stream is not None:
            last = self.last_in.get(stream)
            if last is not None and t < last - 1e-6:
                return False
        return True

    def _accept(self, t, stream):
        last = self.last_in.get(stream)
        self.last_in[stream] = t if last is None else max(t, last)

    def on_wheel(self, t, bogie, v_kmh):
        if bogie not in ('front', 'rear') or not self._valid_time(t, bogie):
            return
        if v_kmh is None or not math.isfinite(v_kmh) or v_kmh < -5 or v_kmh > 150:
            return
        self._accept(t, bogie)
        self._propagate(t)
        z = v_kmh * KMH
        self.wheel[bogie] = (t, z)
        if self.p['use_model'] and self.traction:
            self._update_speed(t, bogie, z)
        else:
            self.v = self._speed(t)
        self._check_stop(t)
        self._check_branch()

    def on_cmd(self, t, notch):
        if (notch is None or not isinstance(notch, (int, float)) or not math.isfinite(notch)
                or not -15 <= notch <= 15 or not self._valid_time(t, 'cmd')):
            return
        self._accept(t, 'cmd')
        self._propagate(t)
        self.notch = int(notch)
        self.cmd_hist.append((t, self.notch))
        lag = self.traction['lag'] if self.traction else 0.0
        while len(self.cmd_hist) > 2 and self.cmd_hist[1][0] <= t - lag:
            self.cmd_hist.pop(0)

    def on_gnss(self, t, antenna, lat, lon, status=2):
        """Фикс GNSS: выставка в окне, дальше (gnss_correction) - мягкая поправка пути."""
        if antenna not in ('master', 'rover') or not self._valid_time(t, antenna):
            return
        if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
            return
        if not self.aligning:
            if self.initialized and self.p['gnss_correction']:
                self._gnss_correct(t, antenna, lat, lon, status)
            return
        if status != 2 or not (math.isfinite(lat) and math.isfinite(lon)) or lat == 0.0:
            return
        if self.gnss_t0 is None:
            self.gnss_t0 = t
        if t - self.gnss_t0 > self.p['init_window']:
            self.aligning = False
            return
        x, y = latlon_to_mgrs_local(lat, lon)
        self._accept(t, antenna)
        self._propagate(t)
        self.last_fix[antenna] = (t, float(x), float(y))
        self.fix_hist.append((self.odo, float(x), float(y),
                              -(MASTER_X if antenna == 'master' else ROVER_X)))
        self._align(t)

    def init_from_xy(self, t, x, y, yaw=None):
        """Ручная выставка без GNSS: base_link в (x, y) MGRS local, курс необязателен."""
        if (not self._valid_time(t) or x is None or y is None
                or not (math.isfinite(x) and math.isfinite(y))
                or (yaw is not None and not math.isfinite(yaw))):
            return
        best = None
        for route in self.routes:
            if route.meta.get('tail_kind') == 'stub' and self._stub_sibling_of(route):
                continue
            s, xt, ryaw = route.project(x, y)
            cost = xt + (100.0 if yaw is not None and math.cos(yaw - ryaw) < 0 else 0.0)
            if best is None or cost < best[0] - 1e-9 or (abs(cost - best[0]) < 1.0 and route.length - s > best[3]):
                best = (cost, route, s, route.length - s)
        if best is None:
            return
        self._propagate(t)
        self.route, self.s = best[1], best[2]
        self.initialized, self.aligning = True, False
        self.var0, self.d_since = self.p['init_std'] ** 2, 0.0
        self.align_odo = self.odo
        self.scale_anchors = ([(self.odo, self.s, max(self.var0, self.p['anchor_init_var']))]
                              if self.p['anchor_scale'] and self.p['anchor_use_init'] else [])

    def diagnostics(self):
        return {
            'initialized': self.initialized, 'aligning': self.aligning,
            'route': self.route.name if self.route else '', 's': self.s,
            'sigma_s': self.sigma_s() if self.initialized else float('nan'),
            'slip': self.slip, 'slip_mode': self.slip_mode, 'n_slip': self.n_slip,
            'n_rejected': self.n_rejected, 'n_corrections': self.n_corrections,
            'n_recoveries': self.n_recoveries, 'n_queue': self.n_queue, 'n_branch': self.n_branch, 'n_gnss_corr': self.n_gnss_corr, 'wheel_scale': self.p['wheel_scale'] * self.scale,
            'notch': self.notch,
        }

    # ----------------------------------------------------------------- выходы
    def state(self, t):
        """(v, x, y, z, yaw, s) на момент t с прогнозом или None до выставки."""
        if not self.initialized:
            return None
        self._propagate(t)
        x, y, z, yaw = self.route.point(self.s)
        return self.v, x, y, z, yaw, self.s

    # ---------------------------------------------------------------- внутреннее
    def model_accel(self, t):
        """Модель тяги: ускорение (м/с^2) по позиции контроллера `lag` с назад и текущей скорости."""
        tr = self.traction
        lag = tr['lag']
        notch = 0
        for tc, n in self.cmd_hist:
            if tc <= t - lag:
                notch = n
            else:
                break
        if self.v < 0.05 and notch <= 0:
            return 0.0
        bins = tr['v_bins']
        j = 0
        while j < len(bins) - 2 and self.v >= bins[j + 1]:
            j += 1
        return tr['accel'][max(min(notch, 15), -15) + 15][j]

    def _wheel_accel(self, t, bogie, z):
        h = self.whist.setdefault(bogie, [])
        h.append((t, z))
        while len(h) > 2 and h[1][0] <= t - self.p['slip_window']:
            h.pop(0)
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.5 * self.p['slip_window']:
            return None
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])

    def _update_speed(self, t, bogie, z):
        """Обновление по тележке. Две согласные тележки принимаются; несогласная проверяется по
        прогнозу модели тяги; при невозможном ускорении колеса (буксование, юз) скорость ведёт
        только модель, пока колёса не вернутся."""
        p = self.p
        z = z * p['wheel_scale']
        r2 = p['wheel_std'] ** 2
        y = z - self.v
        aw = self._wheel_accel(t, bogie, z)
        am = self.model_accel(t)
        others = [w * p['wheel_scale'] for b, (tw, w) in self.wheel.items()
                  if b != bogie and t - tw <= p['wheel_timeout']]
        if self.slip_mode:
            tol = p['slip_exit_tol'] + p['slip_exit_rate'] * (t - self.slip_t0)
            if abs(y) < tol or t - self.slip_t0 > p['recover_time']:
                self.slip_mode = False
            else:
                self.slip = True
                return
        if aw is not None and self.v > 1.0 and (
                abs(aw) > p['slip_accel_max'] or
                (self.notch > 0 and aw > max(am, 0.0) + p['spin_margin'])):
            self.slip_mode, self.slip_t0, self.slip = True, t, True
            self.n_slip += 1
            return
        if others and abs(others[0] - z) > p['agree_tol']:
            # тележки расходятся: берём ту, что ближе к прогнозу
            self.slip = True
            if abs(y) > abs(others[0] - self.v):
                self.n_rejected += 1
                return
        elif not others:
            gate = max(p['gate_min'], p['gate_sigma'] * math.sqrt(self.pv + r2))
            if abs(y) > gate:
                t0 = self.reject_since.setdefault(bogie, t)
                self.n_rejected += 1
                if t - t0 <= p['recover_time']:
                    return
        else:
            self.slip = False
        self.reject_since.pop(bogie, None)
        k = self.pv / (self.pv + r2)
        before = self.v
        self.v = max(self.v + k * y, 0.0)
        self.pv = (1 - k) * self.pv
        knot = self.t_start if self.last_update_t is None else self.last_update_t
        interval = 0.0 if knot is None else max(t - knot, 0.0)
        self.last_update_t = t
        if p['distance_innovation_gain'] and p['along_mode'] != 'pf':
            # Трапеция: апостериорная скорость линейна между принятыми обновлениями, а путь с
            # предыдущего обновления проинтегрирован по прогнозу. Узел - предыдущее принятое
            # показание колеса, поэтому сообщения контроллера, запросы выхода и почти
            # одновременные показания тележек результат не меняют.
            distance = (p['distance_innovation_gain'] * (self.v - before)
                        * min(interval, p['distance_innovation_max_dt']))
            self.odo += distance
            if self.initialized:
                ds = self.scale * distance
                self.s = min(max(self.s + ds, 0.0), self.route.length)
                self.d_since += ds

    def _sibling(self, route, kind):
        """Маршрут, совпадающий с `route` до развилки на конечной, но с другим хвостом."""
        m = route.meta
        for r in self.routes:
            if (r is not route and r.meta.get('direction') == m.get('direction')
                    and r.meta.get('prefix_from') == m.get('prefix_from') and r.meta.get('tail_kind') == kind):
                return r
        return None

    def _stub_sibling_of(self, route):
        return self._sibling(route, 'loop')

    def _gnss_correct(self, t, antenna, lat, lon, status):
        """Поправка s по фиксу GNSS после выставки. Фикс считается шумным (gnss_corr_std), берётся
        только RTK без расхождения антенн, близкий к пути и к прогнозу, не чаще
        gnss_corr_min_dt; скорость не меняется."""
        p = self.p
        if (status != 2 or not (math.isfinite(lat) and math.isfinite(lon)) or self.route is None
                or self.ps is not None):
            return
        x, y = latlon_to_mgrs_local(lat, lon)
        self.gnss_last[antenna] = (t, x, y)
        # статус RTK бывает и у сбойных фиксов: если есть близкий по времени фикс второй антенны,
        # а расстояние между ними не равно базе 12.436 м, приёмник сбоит - фиксы не берём ещё
        # gnss_corr_holdoff (сразу после сбоя они смещены). Одиночный фикс принимается.
        o = self.gnss_last.get('rover' if antenna == 'master' else 'master')
        if (o is not None and abs(o[0] - t) <= p['gnss_corr_pair_dt']
                and abs(math.hypot(o[1] - x, o[2] - y) - BASELINE)
                > p['gnss_corr_pair_tol'] + abs(self.v) * abs(o[0] - t)):
            self.gnss_bad_t = t
            u = self.gnss_undo
            if u is not None and abs(u[0] - t) <= p['gnss_corr_pair_dt']:
                # фикс первой антенны пары уже поправил s - откатываем
                self.s = min(max(self.s - u[1], 0.0), self.route.length)
                self.var0, self.d_since = u[2], u[3] + self.d_since
                self.n_gnss_corr -= 1
            self.gnss_undo = None
        if self.gnss_bad_t is not None and 0.0 <= t - self.gnss_bad_t < p['gnss_corr_holdoff']:
            return
        if self.gnss_corr_t is not None and t - self.gnss_corr_t < p['gnss_corr_min_dt']:
            return
        self._accept(t, antenna)
        self._propagate(t)
        off = -(MASTER_X if antenna == 'master' else ROVER_X)     # s_base = s_антенны + off
        s_fix = self.s - max(self.t - t, 0.0) * self.v * self.scale   # положение на момент фикса
        near = self.route.project_near(x, y, s_fix - off, p['gnss_corr_window'])
        if near is None:
            return
        s_ant, xt = near
        self._gnss_branch(x, y, s_fix - off, xt)
        if self.route.length - (s_ant + off) < 1.0:
            return                  # у конца маршрута проекция упирается в край
        fork = self.route.meta.get('fork_s')
        on_tail = fork is not None and s_ant + off > fork     # за развилкой западной конечной
        if xt > p['gnss_corr_max_xt_tail' if on_tail else 'gnss_corr_max_xt']:
            return
        innov = s_ant + off - s_fix
        sig = self.sigma_s()
        std2 = p['gnss_corr_std'] ** 2
        if abs(innov) > max(p['gnss_corr_gate'], 3.0 * math.sqrt(sig * sig + std2)):
            return
        var = sig * sig
        k = var / (var + std2)
        s0 = self.s
        self.s = min(max(self.s + k * innov, 0.0), self.route.length)
        self.gnss_undo = (t, self.s - s0, self.var0, self.d_since)
        self.var0, self.d_since = (1 - k) * var, 0.0
        self.gnss_corr_t = t
        self.n_gnss_corr += 1

    def _gnss_branch(self, x, y, s_ant, xt):
        """После развилки на конечной: три фикса подряд ближе к другой ветке больше чем на 6 м -
        переход на неё при той же s (соседние ветки у развилки ближе 6 м друг к другу, там
        шумный фикс не решает)."""
        fs = self.route.meta.get('fork_s')
        kind = self.route.meta.get('tail_kind')
        if fs is None or kind not in ('loop', 'stub') or s_ant < fs + 5.0:
            self.gnss_branch_votes = 0
            return
        other = self._sibling(self.route, 'stub' if kind == 'loop' else 'loop')
        near = other.project_near(x, y, s_ant, self.p['gnss_corr_window']) if other else None
        if near is None or xt - near[1] < 6.0:
            self.gnss_branch_votes = 0
            return
        self.gnss_branch_votes += 1
        if self.gnss_branch_votes >= 3:
            self.route = other
            self.branch_by_gnss = True
            self.gnss_branch_votes = 0
            self.n_branch += 1

    def _check_branch(self):
        if self.branch_by_gnss:
            return
        if not self.p['branch_enabled'] or self.route is None or self.route.meta.get('tail_kind') != 'loop':
            return
        fs = self.route.meta.get('fork_s')
        if fs is None or not (fs + self.p['branch_start'] < self.s < fs + self.p['branch_window']) or self.v < self.p['branch_speed']:
            return
        r = self._sibling(self.route, 'stub')
        if r is not None:
            self.route = r              # до развилки варианты совпадают - та же s, отличается только хвост
            self.n_branch += 1
            if self.ps is not None:
                np.clip(self.ps, 0.0, r.length, out=self.ps)

    def sigma_s(self):
        return math.sqrt(self.var0 + (self.p['drift_rate'] * self.d_since) ** 2)

    def _index_stops(self, stops):
        import numpy as np
        for r in self.routes:
            ss, terminal = [], []
            for c in stops:
                if c['n_bags'] < self.p['stop_min_bags'] or c['mad'] > self.p['stop_max_mad']:
                    continue
                s, xt, _ = r.project(c['x'], c['y'])
                if xt < 1.5:
                    ss.append(s)
                    if c.get('direction') == 'terminal':
                        terminal.append(s)
            self.stop_s[r.name] = np.array(sorted(ss))
            self.terminal_s[r.name] = np.array(terminal)

    def _check_stop(self, t):
        standing = (all(v < self.p['stop_speed'] for (tw, v) in self.wheel.values())
                    and self.wheel and self.v < self.p['stop_speed'])
        if not standing:
            self.stand_t0 = None
            self.stand_used = False
            return
        if self.stand_t0 is None:
            self.stand_t0 = t
        if self.stand_used or not self.initialized or t - self.stand_t0 < self.p['stop_dwell']:
            return
        self.stand_used = True
        if self.odo - self.align_odo < self.p['stop_min_travel']:
            return
        cps = self.stop_s.get(self.route.name)
        if cps is None or not len(cps):
            return
        if self.ps is not None:
            self._pf_stop(cps)
            return
        i = int(np.abs(cps - self.s).argmin())
        innov = float(cps[i] - self.s)
        terminals = self.terminal_s.get(self.route.name, np.array([]))
        terminal = bool(len(terminals) and np.min(np.abs(terminals - cps[i])) < 1.0)
        # станции на конечных: только малые поправки и не сразу после выставки
        if terminal and (abs(innov) > 5.0 or self.odo - self.align_odo < 30.0):
            return
        if (self.last_station is not None and abs(cps[i] - self.last_station[0]) < 1.0
                and self.odo - self.last_station[1] < self.p['stop_repeat_dist']):
            return
        sig = self.sigma_s()
        gate = min(self.p['stop_gate_max'], max(self.p['stop_gate_min'], self.p['stop_gate_sigma'] * sig))
        if self.p['stop_bootstrap_gate'] and self.n_corrections == 0 and self.var0 > 100.0:
            gate = min(gate, self.p['stop_bootstrap_gate'])
        qs = self.queue_s.get(self.route.name)
        if qs:
            # сравниваем 'у станции' и 'в известном месте очереди / светофора'
            r2 = self.p['stop_meas_std'] ** 2
            ls = -0.5 * innov * innov / (sig * sig + r2) - 0.5 * math.log(sig * sig + r2)
            best_q = max(((-0.5 * (qq - self.s) ** 2 / (sig * sig + qd * qd) - 0.5 * math.log(sig * sig + qd * qd)
                           + math.log(self.p['queue_prior']), qq, qd) for qq, qd in qs), key=lambda z: z[0])
            if best_q[0] > ls and abs(best_q[1] - self.s) < gate:
                self.n_queue += 1
                if self.p['queue_mode'] == 'soft':
                    var, rq = sig * sig, best_q[2] ** 2
                    k = var / (var + rq)
                    self.s += k * (best_q[1] - self.s)
                    self.var0, self.d_since = (1 - k) * var, 0.0
                return
        if abs(innov) > gate:
            pend = self.pending
            self.pending = (self.odo, innov, float(cps[i]))
            if (pend and 50.0 < self.odo - pend[0] < self.p['recover_window'] and pend[2] != cps[i]
                    and abs(innov) < self.p['recover_max'] and abs(innov - pend[1]) < self.p['recover_agree']):
                # один и тот же сдвиг объясняет две станции подряд: оценка ушла
                self.var0 = max(self.var0, 25.0 + innov * innov)
                self.d_since = 0.0
                self.pending = None
                self.n_recoveries += 1
                sig = self.sigma_s()
            else:
                return
        self.pending = None
        var = sig * sig
        r2 = self.p['stop_meas_std'] ** 2
        k = var / (var + r2)
        if self.p['scale_adapt'] and self.d_since > self.p['scale_min_dist']:
            corr = 1.0 + self.p['scale_gain'] * k * innov / self.d_since
            lim = self.p['scale_limit']
            self.scale = min(max(self.scale * corr, 1.0 - lim), 1.0 + lim)
        self.s += k * innov
        self.var0 = (1 - k) * var
        self.d_since = 0.0
        self.n_corrections += 1
        self.last_innovation = innov
        self.last_station = (float(cps[i]), self.odo)
        if self.p['anchor_scale'] and not terminal:
            self._anchor_scale_update(float(cps[i]))

    def _anchor_scale_update(self, cp):
        """Масштаб пути: взвешенная регрессия s_станции = b + k * odo по принятым станциям,
        ridge-априор на k около 1, свободный сдвиг b поглощает ошибку выставки."""
        if self.scale_anchors and abs(cp - self.scale_anchors[-1][1]) <= 10.0:
            return
        self.scale_anchors.append((self.odo, cp, self.p['anchor_stop_var']))
        self.scale_anchors = self.scale_anchors[-32:]
        if len(self.scale_anchors) < 2:
            return
        a = np.asarray(self.scale_anchors)
        x = a[:, 0] - a[0, 0]
        y = a[:, 1] - a[0, 1] - x
        w = 1.0 / a[:, 2]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        slope = np.sum(w * (x - mx) * (y - my)) / (
            np.sum(w * (x - mx) ** 2) + 1.0 / self.p['anchor_scale_std'] ** 2)
        if abs(slope) < self.p['anchor_deadband']:
            slope = 0.0
        self.scale = 1.0 + float(np.clip(slope, -self.p['scale_limit'], self.p['scale_limit']))

    def _pf_stop(self, cps):
        """Измерение по остановке: смесь 'у известной станции' + 'в другом месте'."""
        p = self.p
        n = len(self.ps)
        m = int(p['pf_explore'] * n)
        if m:
            sd = math.sqrt(float(np.dot(self.pw, (self.ps - self.s) ** 2)))
            half = max(p['pf_explore_min'], 3 * sd)
            j = self.rng.choice(n, m, replace=False)
            self.ps[j] = np.clip(self.s + self.rng.uniform(-half, half, m), 0.0, self.route.length)
        # восстановление: две подряд остановки вне допуска с одинаковым сдвигом
        sd = math.sqrt(float(np.dot(self.pw, (self.ps - self.s) ** 2)))
        i = int(np.abs(cps - self.s).argmin())
        off = float(cps[i] - self.s)
        if abs(off) > 3 * sd + 5.0 and abs(off) < p['recover_max']:
            pend = self.pending
            self.pending = (self.odo, off, float(cps[i]))
            if (pend and 50.0 < self.odo - pend[0] < p['recover_window'] and pend[2] != cps[i]
                    and abs(off - pend[1]) < p['recover_agree']):
                self.ps += off
                self.pending = None
                self.n_recoveries += 1
        else:
            self.pending = None
        near = cps[(cps > self.ps.min() - 50) & (cps < self.ps.max() + 50)]
        sm = p['stop_meas_std']
        lik = np.full(len(self.ps), (1 - p['pf_p_station']) * p['pf_off_density'])
        if len(near):
            d = self.ps[:, None] - near[None, :]
            lik += p['pf_p_station'] * np.exp(-0.5 * (d / sm) ** 2).sum(1) / (math.sqrt(2 * math.pi) * sm)
        w = self.pw * lik
        tot = w.sum()
        if tot <= 0 or not math.isfinite(tot):
            return
        self.pw = w / tot
        self.n_corrections += 1
        ess = 1.0 / np.sum(self.pw ** 2)
        if ess < 0.5 * len(self.pw):
            n = len(self.pw)
            pos = (self.rng.random() + np.arange(n)) / n
            idx = np.minimum(np.searchsorted(np.cumsum(self.pw), pos), n - 1)
            self.ps = self.ps[idx] + self.rng.normal(0, p['pf_s_jitter'], n)
            self.pk = self.pk[idx] + self.rng.normal(0, p['pf_scale_jitter'], n)
            self.pw = np.full(n, 1.0 / n)
        self.s = float(np.dot(self.pw, self.ps))
        self.scale = float(np.dot(self.pw, self.pk))

    def _motion_penalty(self, route):
        """Штраф, если движение по GNSS вдоль маршрута не согласуется с одометром колёс."""
        if len(self.fix_hist) < 2:
            return 0.0
        o0, x0, y0, _ = self.fix_hist[0]
        o1, x1, y1, _ = self.fix_hist[-1]
        if o1 - o0 < 3.0:           # мало движения, направление не определить
            return 0.0
        ds = route.project(x1, y1)[0] - route.project(x0, y0)[0]
        return 0.0 if ds > 0 else 100.0

    def _speed(self, t):
        vs = [v for (tw, v) in self.wheel.values() if t - tw <= self.p['wheel_timeout']]
        if not vs:
            return self.v
        return self.p['wheel_scale'] * sum(vs) / len(vs)

    def _propagate(self, t):
        if self.t is None:
            self.t = self.t_start = t
            return
        dt = t - self.t
        if dt <= 0:
            return
        v0 = self.v
        if self.p['use_model'] and self.traction and dt < 1.0:
            a = self.model_accel(t)
            self.v = min(max(self.v + a * dt, 0.0), 30.0)
            self.pv += (self.p['accel_std'] ** 2) * dt
        v_mid = 0.5 * (v0 + self.v)
        ds = self.scale * v_mid * dt
        self.odo += v_mid * dt
        if self.initialized:
            if self.ps is not None:
                np.clip(self.ps + self.pk * (v_mid * dt), 0.0, self.route.length, out=self.ps)
                self.s = float(np.dot(self.pw, self.ps))
            else:
                self.s = min(max(self.s + ds, 0.0), self.route.length)
            self.d_since += ds
        self.t = t

    def _align(self, t):
        m = self.last_fix.get('master')
        r = self.last_fix.get('rover')
        heading = None
        if m and r and abs(m[0] - r[0]) < 0.06:
            dx, dy = r[1] - m[1], r[2] - m[2]
            L = math.hypot(dx, dy)
            if abs(L - BASELINE) < 1.0:
                ux, uy = dx / L, dy / L
                heading = math.atan2(uy, ux)
                cands = [(m[1] - MASTER_X * ux, m[2] - MASTER_X * uy, 0.0)]
            else:
                cands = [(m[1], m[2], -MASTER_X), (r[1], r[2], -ROVER_X)]
        else:
            cands = [(f[1], f[2], -(MASTER_X if a == 'master' else ROVER_X))
                     for a, f in self.last_fix.items() if abs(f[0] - t) < 0.2]
        cands_r = []
        for x, y, ds in cands:
            for route in self.routes:
                if route.meta.get('tail_kind') == 'stub' and self._stub_sibling_of(route):
                    continue            # вариант с тупиком выбирается только правилом развилки
                projections = (route.project_candidates(x, y, self.p['init_fallback_xt'])
                               if self.p['init_prefix_candidates'] else [route.project(x, y)])
                for s, xt, yaw in projections:
                    if xt > self.p['init_fallback_xt']:
                        continue
                    cost = xt if xt <= self.p['init_max_xt'] else 50.0 + xt
                    if heading is not None and math.cos(heading - yaw) < 0:
                        cost += 100.0
                    cost += self._motion_penalty(route)
                    cands_r.append((cost, route.length - s, route, s + ds))
        if not cands_r:
            return
        # правила выставки для фильтра с одной гипотезой
        if self.p['along_mode'] != 'pf':
            if self.p['init_observed_priority']:
                observed = [c for c in cands_r if not c[2].meta.get('east_osm_extension')]
                bestcost = min(c[0] for c in cands_r)
                usable = [c for c in observed if c[1] > 30.0 and c[0] <= bestcost + 3.0]
                if usable:
                    cands_r = observed
            if self.p['init_end_guard']:
                continuing = [c for c in cands_r if c[1] > 30.0]
                terminal_best = min(cands_r, key=lambda c: c[0])
                if continuing and terminal_best[1] < 15.0:
                    viable = [c for c in continuing if c[0] <= terminal_best[0] + 3.0]
                    if viable:
                        cands_r = viable
        # среди (почти) равно близких вариантов - тот, где больше пути впереди
        cmin = min(c[0] for c in cands_r)
        tie = self.p['init_fallback_tie'] if cmin >= 50.0 and heading is None else 1.0
        best = max((c for c in cands_r if c[0] <= cmin + tie), key=lambda c: c[1])
        best = (best[0], best[2], best[3])
        self._propagate(t)
        self.route, self.s = best[1], min(max(best[2], 0.0), best[1].length)
        self.initialized = True
        self.align_odo = self.odo
        # сомнительная выставка: далеко от всех путей или в самом начале маршрута
        # (трамвай может стоять на пути вне карты, например внутри кольца)
        doubtful = best[0] >= 50.0 or best[2] < self.p['init_edge']
        self.var0 = (self.p['init_fallback_std'] if doubtful else self.p['init_std']) ** 2
        self.d_since = 0.0
        if self.p['anchor_scale']:
            self.scale_anchors = ([(self.odo, self.s, max(self.var0, self.p['anchor_init_var']))]
                                  if self.p['anchor_use_init'] else [])
        if self.p['along_mode'] == 'pf':
            n = self.p['pf_n']
            std = math.sqrt(self.var0)
            self.ps = self.s + self.rng.normal(0, std, n)
            self.pk = 1.0 + self.rng.normal(0, self.p['pf_scale_std'], n)
            self.pw = np.full(n, 1.0 / n)
