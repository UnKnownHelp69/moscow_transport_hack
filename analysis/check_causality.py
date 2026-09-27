"""Проверка причинности: все фиксы GNSS после окна выставки, скорости GNSS и эталонные
положения заменяются мусором; выходы оценщика должны совпасть полностью.
"""
import argparse
import json
import numpy as np
import evaluate as E
import refdata
from tram_odometry.route import load_routes,load_stops,load_traction,load_queues,load_vehicles


def online(d,params):
    est=E.Estimator(load_routes(),params,load_stops(),load_traction(),load_queues())
    out=[]
    for t,k,v in zip(*E.events(d)):
        if k==E.FRONT: est.on_wheel(t,'front',v)
        elif k==E.REAR: est.on_wheel(t,'rear',v)
        elif k==E.CMD: est.on_cmd(t,v)
        else:
            if est.aligning:
                ant='master' if k==E.MFIX else 'rover'
                lat,lon,_=d[ant+'_llh'][int(v)]
                est.on_gnss(t,ant,lat,lon)
            continue
        state=est.state(t)
        if state is not None: out.append((t,*state))
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--params',default='{}')
    p.add_argument('--bags',default='30618_0652866c,30618_27e994fc,30639_92226df0')
    args=p.parse_args()
    for bag in args.bags.split(','):
        d=refdata.load(bag)
        params=json.loads(args.params)
        params.setdefault('wheel_scale',load_vehicles().get(bag.split('_')[0],1.))
        reference=online(d,params)
        changed={k:v.copy() for k,v in d.items()}
        initial=min(d[a+'_t'][0] for a in ['master','rover'] if len(d[a+'_t']))
        for ant in ['master','rover']:
            future=changed[ant+'_t']>initial+params.get('init_window',3.)
            changed[ant+'_llh'][future]=[1.,1.,100000.]
        changed['vel_v'][:]=10000.
        changed['ref_xyz'][:]=-1e6
        actual=online(changed,params)
        assert np.array_equal(reference,actual),bag+' future GNSS leak'
        print(bag, len(actual),'outputs: exact future-GNSS invariance PASS',flush=True)


if __name__=='__main__':main()
