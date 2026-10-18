"""Extras: user-scoped recall quality, page diff for one insert, binary-quantized vectors."""
import json, os, re, sqlite3, time, hashlib, shutil, statistics as st, numpy as np, sqlite_vec
mem=json.load(open("memories.json")); E=np.load("emb.npy"); Q=np.load("qemb.npy"); QI=np.load("qidx.npy"); DIM=384; N=len(mem)
def connect(p):
    c=sqlite3.connect(p); c.enable_load_extension(True); sqlite_vec.load(c); return c
f32=lambda v: np.asarray(v,dtype=np.float32).tobytes()
texts={m["id"]:m["text"] for m in mem}; OUT={}
STOP=set("a an the is are do does did we us our to of for in on and or with what which who whom how when where why any it its should be can that this they them their there".split())
fq=lambda q:" OR ".join(f'"{t}"' for t in re.findall(r"[a-z0-9]+",q.lower()) if t not in STOP)
# 1) user-scoped quality: agent searches only the current user's memories (~200 each)
c=connect("db/C_full.db")
uid={m["id"]:m["user_id"] for m in mem}
def hitk(ids,i): t=texts[mem[QI[i]]["id"]]; return any(texts[x]==t for x in ids)
res={"fts5":0,"vec":0,"hybrid":0}
for i in range(len(QI)):
    u=mem[QI[i]]["user_id"]
    f=[r[0] for r in c.execute("select m.id from memories_fts f join memories m on m.id=f.rowid where memories_fts match ? and m.user_id=? order by bm25(memories_fts) limit 50",(fq(mem[QI[i]]["query"]),u))]
    v=[r[0] for r in c.execute("select rowid from memories_vec where embedding match ? and k=2000 order by distance",(f32(Q[i]),)) if uid[r[0]]==u][:50]
    sc={}
    for L in (f,v):
        for rk,x in enumerate(L): sc[x]=sc.get(x,0)+1/(61+rk)
    h=[x for x,_ in sorted(sc.items(),key=lambda z:-z[1])][:5]
    res["fts5"]+=hitk(f[:5],i); res["vec"]+=hitk(v[:5],i); res["hybrid"]+=hitk(h,i)
OUT["user_scoped_hit_at_5"]={k:v/len(QI) for k,v in res.items()}; print(OUT)
# 2) one insert into the full store: which pages change?
shutil.copy("db/C_full.db","db/C_insert.db"); c2=connect("db/C_insert.db")
def pages(p):
    b=open(p,"rb").read(); return [hashlib.md5(b[i:i+4096]).hexdigest() for i in range(0,len(b),4096)]
before=pages("db/C_insert.db")
m=dict(text="alex prefers short answers in plain english, details only when asked",user_id="user-1",kind="pref")
from fastembed import TextEmbedding
model=TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
t=time.perf_counter(); e=np.array(list(model.embed([m["text"]]))[0],dtype=np.float32); emb_ms=(time.perf_counter()-t)*1000
t=time.perf_counter()
with c2:
    c2.execute("insert into memories(id,user_id,kind,text,created_at) values(10001,?,?,?,1760400000)",(m["user_id"],m["kind"],m["text"]))
    c2.execute("insert into memories_fts(rowid,text) values(10001,?)",(m["text"],))
    c2.execute("insert into memories_vec(rowid,embedding) values(10001,?)",(f32(e),))
ins_ms=(time.perf_counter()-t)*1000
c2.close(); after=pages("db/C_insert.db")
changed=[i+1 for i in range(min(len(before),len(after))) if before[i]!=after[i]]
# map changed page numbers to objects
c3=connect("db/C_insert.db"); own=dict(c3.execute("select pageno,name from dbstat"))
by={}
for p in changed+list(range(len(before)+1,len(after)+1)): by[own.get(p,"?")]=by.get(own.get(p,"?"),0)+1
OUT["one_insert"]=dict(embed_ms=emb_ms, insert_txn_ms=ins_ms, pages_before=len(before), pages_after=len(after), pages_changed=len(changed), new_pages=len(after)-len(before), changed_by_object=by)
# recall the same memory with a paraphrase
q="how does alex like replies?"; qe=np.array(list(model.embed([q]))[0],dtype=np.float32)
t=time.perf_counter(); r=c3.execute("select v.rowid, v.distance, m.text from memories_vec v join memories m on m.id=v.rowid where v.embedding match ? and k=3 order by v.distance",(f32(qe),)).fetchall(); rq=(time.perf_counter()-t)*1000
fr=c3.execute("select rowid, text from memories_fts where memories_fts match ? order by bm25(memories_fts) limit 3",(fq(q),)).fetchall()
OUT["one_recall"]=dict(query=q, vec_top3=r, vec_ms=rq, fts_query=fq(q), fts_top3=fr)
print(json.dumps(OUT["one_insert"]),"\n",OUT["one_recall"])
# 3) binary quantized vectors: 384 bits = 48 bytes, KNN on bits then rerank top 50 with float
p="db/G_bit.db"; os.path.exists(p) and os.remove(p); cg=connect(p)
cg.execute("create virtual table vb using vec0(embedding bit[384])")
cg.executemany("insert into vb(rowid,embedding) values(?, vec_quantize_binary(?))",[(mm["id"],f32(E[i])) for i,mm in enumerate(mem)]); cg.commit(); cg.execute("vacuum")
size=os.path.getsize(p); lat=[];hits=0
for i in range(len(QI)):
    t=time.perf_counter()
    cand=[x[0] for x in cg.execute("select rowid from vb where embedding match vec_quantize_binary(?) and k=50 order by distance",(f32(Q[i]),))]
    s=E[np.array(cand)-1]@Q[i]; top=[cand[j] for j in np.argsort(-s)[:5]]
    lat.append((time.perf_counter()-t)*1000); hits+=hitk(top,i)
OUT["binary_vec0"]=dict(file_bytes=size, bytes_per_memory=size/N, median_ms=st.median(lat), hit_at_5_with_rerank=hits/len(QI))
print(OUT["binary_vec0"])
json.dump(OUT,open("results_extra.json","w"),indent=1,default=str)
