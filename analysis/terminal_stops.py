"""Добавить к станциям повторяющиеся места стоянки на конечных вне карты (только train).

Копирует stops.json, станции и очереди сохраняются. Место на конечной принимается, если оно
встречается минимум в 3 train-прогонах, MAD <= 0.5 м и направление однозначно.
Запуск: python analysis/terminal_stops.py --model <папка модели> --output <папка результата>
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import evaluate  # настройка импортов пакета и модели train по умолчанию
import reader as R
import refdata
from stops_db import stop_events
from tram_odometry.route import Route


def build(model, output, min_bags=3):
    model, output = Path(model), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    for src in model.glob('*.json'):
        if src.resolve() != (output/src.name).resolve():
            shutil.copy2(src, output/src.name)
    data = json.loads((model/'stops.json').read_text())
    maps = [Route(n, R.load_map(n)[:,:3]) for n in R.map_names()]
    samples = []
    for bag in refdata.split()[0]:
        d = refdata.load(bag)
        rt, rp = d['ref_t'], d['ref_xyz']
        for t0, t1 in stop_events(d):
            mask = (rt >= t0) & (rt <= t1)
            if mask.sum() < 3:
                continue
            xyz = np.median(rp[mask], axis=0)
            if min(m.project(*xyz[:2])[1] for m in maps) < 2.0:
                continue
            samples.append(dict(bag=bag, xyz=xyz, dwell=t1-t0))
    # жадные кластеры фиксированного радиуса вокруг самой плотной точки
    left = list(samples)
    additions = []
    while left:
        xy = np.array([s['xyz'][:2] for s in left])
        distance = np.linalg.norm(xy[:,None,:] - xy[None,:,:], axis=2)
        i = int((distance < 2.0).sum(1).argmax())
        member = distance[i] < 2.0
        group = [s for s, yes in zip(left,member) if yes]
        left = [s for s, yes in zip(left,member) if not yes]
        if len({s['bag'] for s in group}) < min_bags:
            continue
        xyz = np.array([s['xyz'] for s in group])
        center = np.median(xyz, axis=0)
        residual = np.linalg.norm(xyz[:,:2]-center[:2],axis=1)
        mad = float(np.median(residual))
        if mad > .5:
            continue
        additions.append(dict(x=float(center[0]), y=float(center[1]), mad=mad,
                              n_bags=len({s['bag'] for s in group}), n=len(group),
                              std=float(np.std(residual)), dwell_med=float(np.median([s['dwell'] for s in group])),
                              direction='terminal', bags=sorted({s['bag'] for s in group})))
    data['stops'].extend(additions)
    data['terminal_source'] = 'train'
    (output/'stops.json').write_text(json.dumps(data,indent=1))
    print(json.dumps(additions,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--min-bags',type=int,default=3)
    args=parser.parse_args()
    build(args.model,args.output,args.min_bags)
