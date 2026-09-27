#!/usr/bin/env bash
# Выгрузить bag-файлы из ветки 'data' в dataset/data/ (как git worktree).
# Повторный запуск подтягивает новые bag.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
git fetch origin data
if [ -e dataset/data/.git ]; then
    git -C dataset/data checkout -q --detach origin/data
else
    git worktree prune
    git worktree add -q --detach dataset/data origin/data
fi
echo "bags in dataset/data: $(find dataset/data -name '*.db3' | wc -l)"
