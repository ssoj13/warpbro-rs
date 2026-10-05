import numpy as np, sys
W=384
def load(n):
    a=np.fromfile(n+'.bin',dtype='<f4').reshape(-1,6)
    code=a[:,0]; mode=np.floor(code/8192+1e-6).astype(int); steps=(code-mode*8192).round().astype(int)
    return dict(mode=mode,steps=steps,t=a[:,1],ratio=a[:,2],n=a[:,3:6])
names={0:'clip miss',1:'threshold',2:'refined',3:'CONE',4:'exhausted miss',5:'left interval'}
V={k:load(k) for k in ['A_256_256','B_1024_256','C_256_1024','D_8192_256','E_1024_1024','F_8192_8192']}
for k,v in V.items():
    c=np.bincount(v['mode'],minlength=6); hits=c[1]+c[2]+c[3]
    print(f"{k:12s} hits {hits:6d} | "+" ".join(f"{names[i]}={c[i]}" for i in range(6)))
A=V['A_256_256']; R=V['F_8192_8192']; Dd=V['D_8192_256']
for ref_name,ref in [('D_8192_256',Dd),('F_8192_8192',R)]:
    print(f"\n--- A cone hits vs {ref_name}")
    cone=A['mode']==3
    rh=np.isin(ref['mode'],[1,2,3])
    print(" cone hits:",cone.sum()," ref hit:",(cone&rh).sum()," ref miss:",(cone&~rh).sum(), " ref cone:",(cone&(ref['mode']==3)).sum())
    m=cone&rh
    dt=np.abs(A['t'][m]-ref['t'][m])
    # eps ~ footprint; use ratio-free: report dt in units of A eps estimate unknown -> raw
    print(" |t_A - t_ref| for cone&refhit: median %.3g p90 %.3g p99 %.3g max %.3g"%tuple(np.percentile(dt,[50,90,99,100])))
    na=A['n'][m]; nr=ref['n'][m]
    cos=np.abs(np.sum(na*nr,1))/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nr,axis=1),1e-12)
    ang=np.degrees(np.arccos(np.clip(cos,0,1)))
    print(" normal angle A vs ref: median %.2f p90 %.2f  >5deg %.1f%%"%(np.median(ang),np.percentile(ang,90),100*(ang>5).mean()))
    th=np.isin(A['mode'],[1,2])&np.isin(ref['mode'],[1,2])
    dt2=np.abs(A['t'][th]-ref['t'][th]); 
    na=A['n'][th]; nr=ref['n'][th]
    cos=np.abs(np.sum(na*nr,1))/np.maximum(np.linalg.norm(na,axis=1)*np.linalg.norm(nr,axis=1),1e-12)
    ang=np.degrees(np.arccos(np.clip(cos,0,1)))
    print(" (control) threshold hits both: n=%d |dt| median %.3g p99 %.3g, angle >5deg %.2f%%"%(th.sum(),np.median(dt2),np.percentile(dt2,99),100*(ang>5).mean()))
print("\ncone ratio in A: median %.2f p90 %.2f max %.2f"%tuple(np.percentile(A['ratio'][A['mode']==3],[50,90,100])))
print("A steps of cone hits: median",np.median(A['steps'][A['mode']==3]))
print("D steps for A-cone pixels: median %d p90 %d p99 %d max %d"%tuple(np.percentile(Dd['steps'][A['mode']==3],[50,90,99,100])))
s1=np.fromfile('A_set1.bin',dtype='<f4').reshape(-1,6); c=A['mode']==3
print("A cone pixels: capped steps median %d, min-clamped (eps/2) median %d of 256"%(np.median(s1[c,0]),np.median(s1[c,1])))
s1d=np.fromfile('D_set1.bin',dtype='<f4').reshape(-1,6)
print("D same pixels: capped median %d, clamped median %d"%(np.median(s1d[c,0]),np.median(s1d[c,1])))
