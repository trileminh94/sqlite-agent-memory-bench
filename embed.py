import json, time, numpy as np, os
from fastembed import TextEmbedding
M="sentence-transformers/all-MiniLM-L6-v2"
mem=json.load(open("memories.json"))
t0=time.perf_counter(); model=TextEmbedding(M); t1=time.perf_counter()
texts=[m["text"] for m in mem]
E=np.array(list(model.embed(texts, batch_size=256)),dtype=np.float32); t2=time.perf_counter()
R=np.random.RandomState(11); idx=R.choice(len(mem),300,replace=False)
Q=np.array(list(model.embed([mem[i]["query"] for i in idx])),dtype=np.float32)
# single-query latency
lat=[]
for i in idx[:50]:
    s=time.perf_counter(); list(model.embed([mem[i]["query"]])); lat.append(time.perf_counter()-s)
np.save("emb.npy",E); np.save("qemb.npy",Q); np.save("qidx.npy",idx)
json.dump(dict(model=M,dim=int(E.shape[1]),load_s=t1-t0,embed_10k_s=t2-t1,per_memory_ms=(t2-t1)/len(texts)*1000,
  single_query_embed_ms_median=float(np.median(lat)*1000),norms=[float(np.linalg.norm(E,axis=1).min()),float(np.linalg.norm(E,axis=1).max())]),open("embed_stats.json","w"),indent=1)
print(open("embed_stats.json").read())
