"""Геометрия восточной конечной из архива OSM (dataset/map/osm_east).

Кольцо OSM совмещается жёстким преобразованием с маршрутами train, входящими на конечную и
выходящими с неё, и присоединяется к ним. Без доступа к сети.
"""
from pathlib import Path
import build_osm_route as O


def fetch():
    import json
    root=Path(O.B.R.REPO)/'dataset/map/osm_east'
    if not (root/'raw.json').exists() or not (root/'provenance.json').exists():
        raise FileNotFoundError('dataset/map/osm_east archive is missing')
    data=json.loads((root/'raw.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    required={236632725,236632736,1232873047,448224397,409746401}
    if not required<={w['id'] for w in data['elements']}:
        raise ValueError('Eastern topology missing from archived OSM data')
    return data,provenance


def extend(model,output):
    import json,shutil
    import numpy as np
    B=O.B
    data,provenance=fetch();ways={w['id']:w for w in data['elements']}
    ids=[236632725,236632736,1232873047,448224397,409746401]
    q=O.osm_xyz(ways,ids)
    model,output=Path(model),Path(output);output.mkdir(parents=True,exist_ok=True)
    raw=json.loads((model/'routes.json').read_text())
    routes=[B.Route(r['name'],r['xyz'],r['meta']) for r in raw['routes']]
    incoming=max((r for r in routes if r.meta['direction']=='tallinskaya_shchukinskaya'),key=lambda r:r.meta['suffix_len'])
    end_s=incoming.meta['map_end']+incoming.meta['suffix_len']
    a=incoming.xyz[(incoming.s>end_s-80)&(incoming.s<=end_s)]
    fits=[]
    for r in routes:
        if r.meta['direction']!='shchukinskaya_tallinskaya' or r.meta.get('tail_kind')=='stub':continue
        p=r.xyz[r.s<min(80,r.meta['prefix_len'])]
        if len(p)<5:continue
        _,fit=O.fit_train_rigid(np.vstack([a,p]),q)
        fits.append((fit['rmse'],r,fit))
    if not fits:raise ValueError('No train-supported eastern fitting geometry')
    fits=[f for f in fits if f[0]<=3 and abs(f[2]['rotation_deg'])<=5 and np.linalg.norm(f[2]['translation_at_origin'])<=10]
    if not fits:
        for source in model.glob('*.json'):
            if source.name in ['routes.json','stops.json','traction.json','vehicles.json']:shutil.copy2(source,output/source.name)
        (output/'east_skipped.txt').write_text('No independently gated eastern fit; original calibration retained.\n')
        print('No safe eastern fit; unchanged model',output,flush=True)
        return
    score,_,fit=min(fits,key=lambda x:x[0])
    print('Eastern alignment',json.dumps(fit),flush=True)
    if score>3 or abs(fit['rotation_deg'])>5 or np.linalg.norm(fit['translation_at_origin'])>10:raise ValueError('Unsafe external-map alignment')
    q=O.transform_public(q,fit);qr=B.Route('osm_east',q)
    infos=[];additions=[]
    for r in routes:
        if r.meta['direction']!='shchukinskaya_tallinskaya':continue
        s,xt,_=qr.project(*r.xyz[0,:2])
        if xt>3 or s<10:continue
        u=np.r_[qr.s[qr.s<s],s]
        ext=np.column_stack([np.interp(u,qr.s,q[:,k]) for k in range(3)])
        w=np.clip(1-(s-u)/20,0,1)
        ext[:,:2]+=w[:,None]*(r.xyz[0,:2]-ext[-1,:2]);ext[:,2]=r.xyz[0,2];ext[-1]=r.xyz[0]
        added=float(B.arc_length(ext)[-1]);meta=dict(r.meta)
        meta['prefix_from']+='+osm_east';meta['east_osm_extension']=added
        for key in ['prefix_len','map_s0','map_end','fork_s','chained_at']:
            if meta.get(key) is not None:meta[key]+=added
        additions.append(dict(name=r.name+'_east',meta=meta,xyz=np.round(B.join(ext,r.xyz),3).tolist()))
        infos.append(dict(route=r.name,added_length=added,join_xt=xt))
    raw['routes'].extend(additions)
    raw['east_osm']=dict(provenance=provenance,way_ids=ids,fit=fit,extensions=infos)
    for p in model.glob('*.json'):
        if p.name!='routes.json':shutil.copy2(p,output/p.name)
    (output/'routes.json').write_text(json.dumps(raw,separators=(',',':')))
    print('Extended',len(additions),'routes',str(output),flush=True)


if __name__=='__main__':
    import argparse,json
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model');p.add_argument('--output');args=p.parse_args()
    if args.model:extend(args.model,args.output)
    else:
        data,provenance=fetch();print(json.dumps(provenance,indent=2))
        for w in data['elements']:print(w['id'],w['nodes'][0],w['nodes'][-1],len(w['nodes']),json.dumps(w.get('tags',{}),ensure_ascii=False))
