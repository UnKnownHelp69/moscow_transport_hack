#!/usr/bin/env bash
# Прогон всех bag из папки через чекер организаторов: метрики по каждому bag и среднее.
#   запуск: src/mos_trap/scripts/run_all_bags.sh <папка с bag> [rate]
#   нужно: собранный workspace чекера (см. ДЛЯ_ЖЮРИ.md), source install/setup.bash
# Логи каждого bag в ./checker_out/<bag>/, сводная таблица в ./checker_out/summary.txt.
set -u
DIR=$1; RATE=${2:-1}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p checker_out
SUM=checker_out/summary.txt
printf "%-22s %12s %12s %12s\n" bag pos_rmse_m pos_max_m vel_rmse_ms > "$SUM"
for BAG in "$DIR"/*/; do
    [ -f "$BAG/metadata.yaml" ] || continue
    NAME=$(basename "$BAG")
    echo "=== $NAME"
    OUT=checker_out/$NAME "$HERE/run_with_checker.sh" "${BAG%/}" "$RATE" > /dev/null 2>&1
    LOG=checker_out/$NAME/metrics.log
    POS=$(grep -a "Position metrics" "$LOG" | tail -1 | grep -o "distance: RMSE=[0-9.]*, max=[0-9.]*")
    VEL=$(grep -a "Velocity metrics" "$LOG" | tail -1 | grep -o "RMSE=[0-9.]*" | head -1)
    PR=$(echo "$POS" | sed -n 's/.*RMSE=\([0-9.]*\),.*/\1/p')
    PM=$(echo "$POS" | sed -n 's/.*max=\([0-9.]*\)/\1/p')
    VR=${VEL#RMSE=}
    printf "%-22s %12s %12s %12s\n" "$NAME" "${PR:-n/a}" "${PM:-n/a}" "${VR:-n/a}" | tee -a "$SUM"
done
awk 'NR > 1 && $2 != "n/a" {p += $2; v += $4; n++; if ($3 > m) m = $3}
     END {if (n) printf "%-22s %12.3f %12.3f %12.3f\n", "среднее (" n " bag)", p / n, m, v / n}' "$SUM" | tee -a "$SUM"
