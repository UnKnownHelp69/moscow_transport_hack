#!/usr/bin/env bash
# Всё одной командой в workspace чекера организаторов: наш пакет сообщений (из датасета, с
# DriverControllerCommand) вместо пакета чекера, сборка, прогон всех bag из папки через чекер,
# сводная таблица. <WS> - корень workspace чекера (в их Docker-контейнере /workspace).
#   структура: <WS>/src/{checker_ros, tram_vehicle_msgs, mos_trap}, bag в <WS>/bags/
#   запуск:    cd <WS> && src/mos_trap/scripts/run_jury.sh [папка с bag, по умолчанию bags] [rate]
# Логи каждого bag в ./checker_out/<bag>/, таблица в ./checker_out/summary.txt.
set -eu
BAGS=${1:-bags}; RATE=${2:-1}
REPO=$(cd "$(dirname "$0")/.." && pwd)
[ -d "$BAGS" ] || { echo "нет папки с bag: $BAGS"; exit 1; }
set +u; source /opt/ros/humble/setup.bash; set -u
# пакет сообщений чекера исключаем из сборки, наш включаем
for d in src/*/; do
    d=${d%/}
    if [ -f "$d/package.xml" ] && grep -q "<name>tram_vehicle_msgs</name>" "$d/package.xml" \
            && [ "$(cd "$d" && pwd)" != "$REPO/tram_vehicle_msgs" ]; then
        touch "$d/COLCON_IGNORE"
        echo "пакет сообщений $d исключён из сборки, используется $REPO/tram_vehicle_msgs"
    fi
done
rm -f "$REPO/tram_vehicle_msgs/COLCON_IGNORE"
rm -rf build/tram_vehicle_msgs install/tram_vehicle_msgs   # старая сборка из другого каталога
colcon build
set +u; source install/setup.bash; set -u
"$REPO/scripts/run_all_bags.sh" "$BAGS" "$RATE"
