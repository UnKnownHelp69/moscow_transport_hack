"""Поправка высоты официальной карты по train-прогонам.

Для каждого окна 50 м центрлинии: медиана невязки GNSS в каждом прогоне, медиана по прогонам
(не меньше 4), сглаживание по 5 окнам, ограничение +-1 м. К z карты прибавляется половина
(`weight`). Геометрия вне официальной карты не меняется.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import refdata
import reader as R
import build_route as B
from tram_odometry.route import Route

ROOT = Path(__file__).resolve().parents[1]


def fit(bags):
    fields={}
    for name in R.map_names():
        route=Route(name,R.load_map(name)[:,:3]);grid=np.arange(0,route.length+50,50)
        samples=[]
        for bag in bags:
            d=refdata.load(bag);xyz=d['ref_xyz'][::10]
            if len(xyz)<5:continue
            proj=B.project_many(route,xyz)
            s,xt=proj.T
            motion=np.gradient(xyz[:,:2],axis=0);speed=np.linalg.norm(motion,axis=1)
            idx=np.clip(np.searchsorted(route.s,s,side='right')-1,0,len(route.s)-2)
            tangent=route._d[idx]/np.sqrt(route._l2[idx,None])
            heading=(motion*tangent).sum(1)/np.maximum(speed,1e-9)
            near=(xt<5)&(s>10)&(s<route.length-10)&(heading>.5)&(speed>.5)
            predicted=np.column_stack([np.interp(s,route.s,route.xyz[:,k]) for k in range(3)])
            delta=xyz-predicted
            cross=-tangent[:,1]*delta[:,0]+tangent[:,0]*delta[:,1]
            bins=np.full((len(grid),2),np.nan)
            for j,g in enumerate(grid):
                use=near&(abs(s-g)<50)
                if use.sum()>=3:bins[j]=np.median(np.column_stack([cross[use],delta[use,2]]),axis=0)
            samples.append(bins)
        a=np.array(samples);values=np.zeros((len(grid),2));counts=np.isfinite(a[:,:,0]).sum(0)
        for j in range(len(grid)):
            if counts[j]>=4:values[j]=np.nanmedian(a[:,j,:],axis=0)
        # сглаживание по 5 окнам; окна без данных остаются нулевыми
        for col in range(2):
            values[:,col]=np.convolve(np.pad(values[:,col],(2,2),mode='edge'),np.ones(5)/5,mode='valid')
        values[counts<4]=0
        values=np.clip(values,-1.,1.)
        fields[name]=dict(grid=grid.tolist(),residuals=values.tolist(),counts=counts.tolist())
    return fields


def build(model,output,weight=.5,xy=False,z=True):
    model,output=Path(model),Path(output);output.mkdir(parents=True,exist_ok=True)
    metadata_path=model/'fold.json'
    bags=json.loads(metadata_path.read_text())['fitting'] if metadata_path.exists() else refdata.split()[0]
    fields=fit(bags)
    raw=json.loads((model/'routes.json').read_text())
    maps={name:Route(name,R.load_map(name)[:,:3]) for name in R.map_names()}
    for r in raw['routes']:
        xyz=np.asarray(r['xyz'],float);meta=r['meta'];name=meta['direction'];route=Route(r['name'],xyz)
        f=fields[name];m=maps[name]
        use=(route.s>=meta['map_s0'])&(route.s<=meta['map_end'])
        s=route.s[use]-meta['map_s0']
        idx=np.clip(np.searchsorted(m.s,s,side='right')-1,0,len(m.s)-2)
        tangent=m._d[idx]/np.sqrt(m._l2[idx,None])
        res=np.array(f['residuals']);cross=np.interp(s,f['grid'],res[:,0]);dz=np.interp(s,f['grid'],res[:,1])
        taper=np.clip(np.minimum(s,m.length-s)/30.,0,1)*weight
        if xy:
            xyz[use,0]-=tangent[:,1]*cross*taper;xyz[use,1]+=tangent[:,0]*cross*taper
        if z:xyz[use,2]+=dz*taper
        # при изменении XY пересчитать метаданные длины дуги
        new_s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz[:,:2],axis=0),axis=1))]
        for key in ['prefix_len','map_s0','map_end','fork_s','chained_at']:
            if meta.get(key) is not None:meta[key]=float(np.interp(meta[key],route.s,new_s))
        r['xyz']=xyz.round(3).tolist()
    raw['research_bias']=dict(training=bags,weight=weight,xy=xy,z=z,fields=fields)
    for p in model.glob('*.json'):
        if p.name!='routes.json':shutil.copy2(p,output/p.name)
    (output/'routes.json').write_text(json.dumps(raw,separators=(',',':')))
    print('Map residual calibration',output,'bags',len(bags),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--weight',type=float,default=.5);p.add_argument('--xy',action='store_true');p.add_argument('--no-z',action='store_true');a=p.parse_args()
    build(a.model,a.output,a.weight,a.xy,not a.no_z)
