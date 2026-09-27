#!/usr/bin/env bash
# Запуск нашего узла с чекером организаторов (check-code) в его workspace или контейнере.
#   Структура: /workspace/src/{checker_ros, tram_vehicle_msgs, mos_trap}   (mos_trap - этот репозиторий)
#   Сборка:    cd /workspace && colcon build && source install/setup.bash
#   Запуск:    src/mos_trap/scripts/run_with_checker.sh bags/30618_88aea4d9 [rate] [vehicle_id]
#   vehicle_id по умолчанию - префикс имени bag (30618 / 30639), задаёт масштаб колёс вагона.
# Печатает итоговые строки RMSE/max чекера; логи в ./checker_out/ (или в $OUT).
set -u
BAG=$1; RATE=${2:-1}; NAME=$(basename "$BAG"); VEH=${3:-${NAME%%_*}}
OUT=${OUT:-checker_out}; mkdir -p "$OUT"
CHK=$(ros2 pkg prefix hackathon_solution_checker)/lib/hackathon_solution_checker/metrics
ODO=$(ros2 pkg prefix tram_odometry)
"$CHK" --ros-args -p report_period_sec:=60.0 > "$OUT/metrics.log" 2>&1 &
M=$!
"$ODO/lib/tram_odometry/tram_odometry_node" --ros-args \
    --params-file "$ODO/share/tram_odometry/config/params.yaml" -p vehicle_id:="'$VEH'" > "$OUT/node.log" 2>&1 &
N=$!
sleep 4
ros2 bag play "$BAG" --rate "$RATE"
sleep 3
kill -INT $M; wait $M
kill -INT $N; wait $N
grep -aE "Velocity metrics|Position metrics" "$OUT/metrics.log" | tail -2
