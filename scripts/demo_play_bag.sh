#!/usr/bin/env bash
# Демо для жюри: узел одометрии + монитор точности, проигрывание одного bag, итоговые метрики.
#   запуск: scripts/demo_play_bag.sh <папка bag> [vehicle_id] [rate]
#   нужно: ROS 2 Humble и собранный workspace (colcon build; source install/setup.bash)
# Результат в ./demo_out/<bag>/: node.log (раз в секунду: частота, задержка, sigma, юз),
# eval.log (текущие и итоговые FINAL метрики), matched.csv (эталон и оценка по точкам).
set -u
BAG=$1; VEH=${2:-}; RATE=${3:-1}
NAME=$(basename "$BAG"); OUT=demo_out/$NAME; mkdir -p "$OUT"
[ -z "$VEH" ] && VEH=${NAME%%_*}
ros2 run tram_odometry tram_odometry_node --ros-args \
    --params-file "$(ros2 pkg prefix tram_odometry)/share/tram_odometry/config/params.yaml" \
    -p vehicle_id:="'$VEH'" > "$OUT/node.log" 2>&1 &
N=$!
ros2 run tram_odometry tram_odometry_eval --ros-args -p csv_path:="$PWD/$OUT/matched.csv" > "$OUT/eval.log" 2>&1 &
E=$!
sleep 3
ros2 bag play "$BAG" --rate "$RATE"
sleep 3
pkill -INT -P $E; kill -INT $E 2>/dev/null; sleep 3
pkill -INT -P $N; kill -INT $N 2>/dev/null; sleep 1
grep -a FINAL "$OUT/eval.log"
echo "logs: $OUT/"
