# Атрибуция OpenStreetMap

Геометрия западной и восточной конечных в `routes.json` содержит данные © участники
OpenStreetMap, лицензия Open Database License (ODbL) 1.0:
https://www.openstreetmap.org/copyright, https://opendatacommons.org/licenses/odbl/1-0/

| конечная | выгрузка | время (UTC) | архив |
|---|---|---|---|
| западная | https://api.openstreetmap.org/api/0.6/map?bbox=37.385,55.797,37.394,55.802 | 2026-09-26T13:17:03 | `dataset/map/osm_west/` |
| восточная | https://api.openstreetmap.org/api/0.6/map?bbox=37.456,55.807,37.466,55.814 | 2026-09-26T15:21:25 | `dataset/map/osm_east/` |

ID путей восточной конечной: 236632725, 236632736, 1232873047, 448224397, 409746401.
В `routes.json` записаны ID и версии путей и жёсткое преобразование, совмещающее OSM с
маршрутами train. Пересборка: `analysis/build_model.py` (`build_osm_route.py`, `east_map.py`).
