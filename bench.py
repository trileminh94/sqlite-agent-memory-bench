"""Agent-memory-in-one-SQLite-file benchmark. Local only, no network, no paid APIs."""
import json, os, re, sqlite3, struct, time, statistics as st, numpy as np, sqlite_vec, platform
mem=json.load(open("memories.json")); E=np.load("emb.npy"); Q=np.load("qemb.npy"); QI=np.load("qidx.npy")
N=len(mem); DIM=E.shape[1]; OUT={}
os.makedirs("db",exist_ok=True)
def connect(p, fresh=True):
    if fresh and os.path.exists(p): os.remove(p)
    c=sqlite3.connect(p); c.enable_load_extension(True); sqlite_vec.load(c); c.enable_load_extension(False); return c
def f32(v): return np.asarray(v,dtype=np.float32).tobytes()
def dbstat(c):
    rows=c.execute("select name, count(*), sum(pgsize), sum(payload), sum(unused), sum(ncell), sum(pagetype='overflow'), sum(pagetype='leaf'), sum(pagetype='internal') from dbstat group by name order by 3 desc").fetchall()
    return [dict(name=r[0],pages=r[1],bytes=r[2],payload=r[3],unused=r[4],cells=r[5],overflow_pages=r[6],leaf_pages=r[7],interior_pages=r[8]) for r in rows]
def finish(c,p):
    c.commit(); c.execute("vacuum"); c.commit()
    pc=c.execute("pragma page_count").fetchone()[0]; ps=c.execute("pragma page_size").fetchone()[0]
    return dict(file_bytes=os.path.getsize(p), page_size=ps, page_count=pc, bytes_per_memory=os.path.getsize(p)/N, objects=dbstat(c))
SCHEMA_ROWS="create table memories(id integer primary key, user_id text not null, kind text not null, text text not null, created_at integer not null)"
def ins_rows(c): c.executemany("insert into memories values(?,?,?,?,?)",[(m["id"],m["user_id"],m["kind"],m["text"],m["created_at"]) for m in mem])

# A: rows only
p="db/A_rows.db"; c=connect(p); c.execute(SCHEMA_ROWS); ins_rows(c); OUT["A_rows_only"]=finish(c,p); c.close()
# B: rows + FTS5 (external content, so text is stored once)
p="db/B_rows_fts5.db"; c=connect(p); c.execute(SCHEMA_ROWS); ins_rows(c)
c.execute("create virtual table memories_fts using fts5(text, content='memories', content_rowid='id')")
c.execute("insert into memories_fts(rowid,text) select id,text from memories"); OUT["B_rows_fts5"]=finish(c,p); c.close()
# C: rows + FTS5 + sqlite-vec vec0 (the full agent-memory store)
p="db/C_full.db"; c=connect(p); c.execute(SCHEMA_ROWS); ins_rows(c)
c.execute("create virtual table memories_fts using fts5(text, content='memories', content_rowid='id')")
c.execute("insert into memories_fts(rowid,text) select id,text from memories")
c.execute(f"create virtual table memories_vec using vec0(embedding float[{DIM}])")
t=time.perf_counter()
c.executemany("insert into memories_vec(rowid,embedding) values(?,?)",[(m["id"],f32(E[i])) for i,m in enumerate(mem)])
vec_insert_s=time.perf_counter()-t
OUT["C_rows_fts5_vec0"]=finish(c,p); OUT["C_rows_fts5_vec0"]["vec0_bulk_insert_s"]=vec_insert_s; c.close()
# D: rows + raw float32 BLOB column (no extension; brute force)
p="db/D_blob.db"; c=connect(p); c.execute(SCHEMA_ROWS.replace("created_at integer not null","created_at integer not null, embedding blob not null"))
c.executemany("insert into memories values(?,?,?,?,?,?)",[(m["id"],m["user_id"],m["kind"],m["text"],m["created_at"],f32(E[i])) for i,m in enumerate(mem)])
OUT["D_rows_blob_f32"]=finish(c,p); c.close()
# E: vector as JSON text inside a row (the way many ORM-based stores do it)
p="db/E_json.db"; c=connect(p); c.execute("create table vectors(id text primary key, vector text not null, metadata text not null default '{}')")
c.executemany("insert into vectors values(?,?,?)",[(f"session-{m['id']}",json.dumps([float(x) for x in E[i]]),json.dumps(dict(text=m["text"],user_id=m["user_id"]))) for i,m in enumerate(mem)])
OUT["E_json_text_vector"]=finish(c,p); c.close()

# single-vector sizes
v=E[0]; OUT["one_vector"]=dict(float32_blob_bytes=len(f32(v)), json_text_bytes=len(json.dumps([float(x) for x in v])),
    json_text_bytes_rounded6=len(json.dumps([round(float(x),6) for x in v])))
# SQLite max local payload on a table leaf page with 4096-byte pages: U-35 = 4061 bytes; min local = (U-12)*32/255-23
U=4096; OUT["page_math"]=dict(max_local_table_leaf=U-35, min_local=(U-12)*32//255-23)

# --- latency + recall ---
STOP=set("a an the is are do does did we us our to of for in on and or with what which who whom how when where why any it its should be can that this they them their there".split())
def fts_q(q):
    toks=[t for t in re.findall(r"[a-z0-9]+",q.lower()) if t not in STOP]
    return " OR ".join(f'"{t}"' for t in toks) or '""'
def timeit(fn, reps):
    ts=[]; res=None
    for i in range(reps):
        s=time.perf_counter(); r=fn(i); ts.append((time.perf_counter()-s)*1000)
        if res is None: res=[]
        res.append(r)
    ts.sort(); return dict(median_ms=st.median(ts), p95_ms=ts[int(.95*len(ts))-1], mean_ms=st.mean(ts)), res
c=connect("db/C_full.db",fresh=False)
texts={m["id"]:m["text"] for m in mem}
K=5; nq=len(QI)
def fts(i,k=K): return [r[0] for r in c.execute("select rowid from memories_fts where memories_fts match ? order by bm25(memories_fts) limit ?",(fts_q(mem[QI[i]]["query"]),k))]
def vec(i,k=K): return [r[0] for r in c.execute("select rowid from memories_vec where embedding match ? and k = ? order by distance",(f32(Q[i]),k))]
def hybrid(i,k=K,depth=50,kk=60):
    sc={}
    for rank,r in enumerate(fts(i,depth)): sc[r]=sc.get(r,0)+1/(kk+rank+1)
    for rank,r in enumerate(vec(i,depth)): sc[r]=sc.get(r,0)+1/(kk+rank+1)
    return [r for r,_ in sorted(sc.items(),key=lambda x:-x[1])[:k]]
cD=connect("db/D_blob.db",fresh=False)
def blob_sql_scan(i,k=K):  # read every blob from SQLite each query, cosine in numpy
    rows=cD.execute("select id, embedding from memories").fetchall()
    M=np.frombuffer(b"".join(r[1] for r in rows),dtype=np.float32).reshape(-1,DIM)
    s=M@Q[i]; top=np.argpartition(-s,k)[:k]; return [rows[j][0] for j in top[np.argsort(-s[top])]]
cE=connect("db/E_json.db",fresh=False)
def json_scan(i,k=K):  # SELECT * then json decode + cosine in app code
    rows=cE.execute("select id, vector, metadata from vectors").fetchall()
    M=np.array([json.loads(r[1]) for r in rows],dtype=np.float32)
    s=M@Q[i]; top=np.argpartition(-s,k)[:k]; return [int(rows[j][0].split("-")[1]) for j in top[np.argsort(-s[top])]]
def hit(results):
    h=0
    for i,r in enumerate(results):
        tgt=texts[mem[QI[i]]["id"]]
        if any(texts[x]==tgt for x in r): h+=1
    return h/len(results)
LAT={}
for name,fn,reps in [("fts5_bm25",fts,nq),("vec0_knn",vec,nq),("hybrid_rrf",hybrid,nq),("blob_scan_numpy",blob_sql_scan,50),("json_text_scan",json_scan,10)]:
    for i in range(3): fn(i)  # warm
    lat,res=timeit(fn,reps); lat["hit_at_5"]=hit(res); lat["queries"]=reps; LAT[name]=lat
    print(name,lat,flush=True)
OUT["latency_k5"]=LAT

# user-scoped recall: filter AFTER global top-5 (a common pattern) vs filter BEFORE (vec0 metadata col)
p="db/F_vec_partition.db"; cf=connect(p); cf.execute(f"create virtual table mv using vec0(user_id text partition key, embedding float[{DIM}])")
cf.executemany("insert into mv(rowid,user_id,embedding) values(?,?,?)",[(m["id"],m["user_id"],f32(E[i])) for i,m in enumerate(mem)]); cf.commit()
post=[];pre=[];prelat=[]
for i in range(nq):
    u=mem[QI[i]]["user_id"]
    g=vec(i,5); post.append(sum(1 for x in g if mem[x-1]["user_id"]==u))
    s=time.perf_counter(); r=cf.execute("select rowid from mv where embedding match ? and k=5 and user_id=?",(f32(Q[i]),u)).fetchall(); prelat.append((time.perf_counter()-s)*1000); pre.append(len(r))
OUT["user_scoped"]=dict(users=len({m["user_id"] for m in mem}), post_filter_avg_results_of_5=st.mean(post), post_filter_zero_result_rate=sum(1 for x in post if x==0)/nq,
    pre_filter_avg_results_of_5=st.mean(pre), pre_filter_median_ms=st.median(prelat))
print(OUT["user_scoped"])

# single memory insert walk: fresh full store, insert one memory, count pages touched (page_count delta + changed pages via dbstat checksum)
OUT["machine"]=dict(cpu="Intel(R) Xeon(R) Processor (8 vCPU VM)", ram_gb=15, python=platform.python_version(), sqlite=sqlite3.sqlite_version, sqlite_vec=c.execute("select vec_version()").fetchone()[0], os=platform.platform())
json.dump(OUT,open("results.json","w"),indent=1)
