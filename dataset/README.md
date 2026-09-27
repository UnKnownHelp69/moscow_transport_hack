# Данные

## Что лежит в папке

| путь | что это |
|---|---|
| `map/shchukinskaya_tallinskaya.json`, `map/tallinskaya_shchukinskaya.json` | официальные центрлинии пути по двум направлениям, MGRS 37U local (`x = UTM37N_E - 300000`, `y = UTM37N_N - 6100000`), шаг ~1 м: `x, y, z, tang, curv` |
| `map/osm_west/`, `map/osm_east/` | архив выгрузок OpenStreetMap для достройки колец на конечных (ODbL) |
| `data/` | bag-файлы, в git не хранятся: `scripts/fetch_data.sh` выгружает их из ветки `data` |
| `cache/` | кэш офлайн-скриптов `analysis/`, в git не хранится |

## Записи

122 записи ROS 2 bag (rosbag2, sqlite3), из них 25 пар побайтных дубликатов, итого 97
уникальных. Два вагона: `30618_*` (103 записи) и `30639_*` (19). Суммарно ~36.7 ч, одна запись
от 1 с до 32 мин, медиана ~20 мин.

| топик | тип | в решении |
|---|---|---|
| `/vehicle/front_bogie_velocity`, `/vehicle/rear_bogie_velocity` | `tram_vehicle_msgs/VelocitySensor` | вход: скорость тележки, км/ч |
| `/vehicle/driver_position_cmd` | `tram_vehicle_msgs/DriverControllerCommand` | вход: позиция контроллера, `0` - нейтраль, `+1…+15` тяга, `-1…-15` тормоз |
| `/sensing/gnss/{master,rover}/fix` | `sensor_msgs/NavSatFix` | только выставка в первые 3 с и эталон для метрик |
| `/sensing/gnss/{master,rover}/vel` | `geometry_msgs/TwistStamped` | только эталон скорости для метрик |

Эталон положения - `base_link` (ось передней тележки на уровне рельса), восстановленный из
двух антенн по tf: `master + 9.873 · unit(rover - master)`, `z = alt - 3`.

## Как использовались

- 64 уникальные записи с GNSS поделены пополам по имени: train (калибровка модели) и val
  (оценка точности), см. `analysis/refdata.py`.
- Калибровка (маршруты, станции, модель тяги, масштаб колёс) строится только по train:
  `analysis/build_model.py`.
