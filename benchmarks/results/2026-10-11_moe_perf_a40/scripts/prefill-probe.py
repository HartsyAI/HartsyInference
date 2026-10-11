import json,subprocess,time,urllib.request,sys
variant,model,pf=sys.argv[1:4]
prompt=open(pf).read()
p=subprocess.Popen(["/root/baseline/bin/hartsy-serve-"+variant,"-m",model,"--port","8098"],stdout=open("/root/baseline/results/probe-"+variant+".log","w"),stderr=subprocess.STDOUT)
for i in range(200):
    try: urllib.request.urlopen("http://127.0.0.1:8098/health",timeout=2); break
    except Exception: time.sleep(3)
def ask(n):
    b={"model":"x","messages":[{"role":"user","content":prompt}],"max_tokens":n,"temperature":float(sys.argv[5]) if len(sys.argv)>5 else 0,"top_k":20,"top_p":0.95,"seed":1,"stream":False,"cache_prompt":False}
    r=urllib.request.Request("http://127.0.0.1:8098/v1/chat/completions",json.dumps(b).encode(),{"Content-Type":"application/json"})
    t=json.load(urllib.request.urlopen(r,timeout=900))["timings"]
    print(variant,"prompt_ms",round(t["prompt_ms"]),"pp_tps",round(t["prompt_per_second"]),"tg_tps",round(t.get("predicted_per_second",0),1),flush=True)
N=int(sys.argv[4]) if len(sys.argv)>4 else 1
for i in range(3): ask(N)
p.terminate(); p.wait(60)
