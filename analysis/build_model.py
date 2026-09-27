"""Сборка данных поставляемой модели из train-прогонов и архивов OSM.

    python3 analysis/refdata.py                  # один раз: кэш эталонов и разбиение train/val
    python3 analysis/build_model.py --output dataset/cache/model

Шаги: build_base.py (маршруты, станции, модель тяги, масштаб колёс) -> кольцо восточной
конечной по OSM (east_map.py) -> поправка высоты карты (map_height.py). Каждое чтение
эталона проверяется по списку train, сеть запрещена. Результат совпадает с
tram_odometry/tram_odometry/data (манифест с SHA-256 в папке результата).
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
FILES = ['routes.json', 'stops.json', 'traction.json', 'vehicles.json']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='dataset/cache/model')
    out = (ROOT / parser.parse_args().output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    for terminal in ['west', 'east']:
        for name in ['raw.json', 'provenance.json']:
            if not (ROOT / 'dataset/map' / ('osm_' + terminal) / name).exists():
                raise RuntimeError('OSM archive missing: osm_%s/%s' % (terminal, name))
    (ROOT / 'dataset/cache').mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='model_', dir=ROOT / 'dataset/cache') as tmp:
        tmp = Path(tmp)
        base, east, height = tmp / 'base', tmp / 'east', tmp / 'height'
        subprocess.run([sys.executable, 'analysis/build_base.py', '--output', str(base)], cwd=ROOT, check=True)
        import refdata
        import east_map
        import map_height
        train = refdata.split()[0]
        load, urlopen = refdata.load, urllib.request.urlopen
        seen = set()

        def train_only(bag):
            if bag not in train:
                raise AssertionError('non-train reference requested: ' + bag)
            seen.add(bag)
            return load(bag)

        def no_network(*args, **kwargs):
            raise AssertionError('network access during model build')

        refdata.load, urllib.request.urlopen = train_only, no_network
        try:
            east_map.extend(base, east)
            map_height.build(base, height, weight=.5, xy=False, z=True)
        finally:
            refdata.load, urllib.request.urlopen = load, urlopen
        # геометрия восточной конечной + исправленные высоты участка карты
        raw = json.loads((east / 'routes.json').read_text())
        heights = json.loads((height / 'routes.json').read_text())
        by_name = {r['name']: r for r in heights['routes']}
        for r in raw['routes']:
            if r['name'] in by_name:
                r['xyz'] = by_name[r['name']]['xyz']
        raw['research_bias'] = heights['research_bias']
        (out / 'routes.json').write_text(json.dumps(raw, separators=(',', ':')))
        for name in FILES[1:]:
            shutil.copy2(base / name, out / name)
    manifest = dict(train_bags=train, reference_bags_read=sorted(seen), offline=True,
                    sha256={n: hashlib.sha256((out / n).read_bytes()).hexdigest() for n in FILES})
    (out / 'rebuild_manifest.txt').write_text(json.dumps(manifest, indent=2))
    print('model:', out, flush=True)


if __name__ == '__main__':
    main()
