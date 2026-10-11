import sys, collections, csv
rows=[]
for line in open(sys.argv[1]):
    p=line.rstrip("\n").split(",",9)
    if len(p)<10: continue
    rows.append((int(p[0]),int(p[1]),int(p[2]),int(p[5]),p[9]))
rows.sort()
# decode steps end at the lm_head matvec: the kernel with the largest grid in each step
big=[i for i,r in enumerate(rows) if r[2]>=20000 and "mul_mat_vec" in r[4]]
print("kernels",len(rows),"lm_head-like launches",len(big))
if len(big)<8: sys.exit("not enough steps")
steps=[]
for a,b in zip(big[:-1],big[1:]):
    steps.append(rows[a+1:b+1])
steps=steps[-20:]   # steady-state decode steps
def stats(step):
    wall=(step[-1][1]-step[0][0])/1e3
    busy=sum(r[1]-r[0] for r in step)/1e3
    return wall,busy,len(step)
w=[stats(s) for s in steps]
print("steps analysed",len(steps),"mean wall us %.0f  mean kernel-busy us %.0f  mean kernels %.0f"%(sum(x[0] for x in w)/len(w),sum(x[1] for x in w)/len(w),sum(x[2] for x in w)/len(w)))
agg=collections.defaultdict(lambda:[0,0])
for s in steps:
    for r in s:
        a=agg[r[4]]; a[0]+=1; a[1]+=r[1]-r[0]
print("%-58s %7s %9s %9s"%("kernel (per decode step)","calls","total us","avg us"))
n=len(steps)
for k,(c,t) in sorted(agg.items(),key=lambda kv:-kv[1][1])[:int(sys.argv[2]) if len(sys.argv)>2 else 28]:
    print("%-58s %7.1f %9.1f %9.2f"%(k[:58],c/n,t/1e3/n,t/1e3/c))
