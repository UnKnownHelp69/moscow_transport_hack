"""Базовая калибровка по train-прогонам: маршруты (+ западное кольцо по OSM), станции,
таблица тяги, масштаб колёс вагонов, места стоянки на конечных. Вызывается из build_model.py.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='dataset/cache/model_candidate')
    parser.add_argument('--loop',choices=['inner','outer'],default='outer')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=(root/args.output).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if not (root/'dataset/cache/split.csv').exists():
        raise SystemExit('Prepare fixed split first: python3 analysis/refdata.py')
    env=dict(os.environ,TRAM_ODOM_DATA=str(out),OPENBLAS_NUM_THREADS='1')
    commands=[['analysis/build_osm_route.py','--topology',args.loop,'--extend-prefixes','--output',str(out/'routes.json')],
              ['analysis/stops_db.py','train'],['analysis/fit_traction.py','train'],
              ['analysis/vehicle_scale.py','train'],
              ['analysis/terminal_stops.py','--model',str(out),'--output',str(out)]]
    for command in commands:
        print('+',sys.executable,*command,flush=True)
        subprocess.run([sys.executable,*command],cwd=root,env=env,check=True)
    print('Candidate model:',out)


if __name__=='__main__':
    main()
