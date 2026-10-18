"""Generate N synthetic but realistic agent memories + paraphrased recall queries.
Deterministic (seed 7). Each memory has a 'query' phrased differently from the memory text,
so keyword search can't always just copy words."""
import random, json, sys
R = random.Random(7)
people = ["alex","sam","river","quinn","jordan","casey","morgan","riley","avery","parker","rowan","skyler","drew","blake","reese","emery","harper","logan","sage","finley"]
projects = ["checkout-service","mobile-app","landing-page","billing-api","search-index","onboarding-flow","video-renderer","auth-gateway","analytics-pipeline","docs-site","storyboard-editor","export-worker"]
dbs = ["postgres","sqlite","mysql","redis","dynamodb","duckdb","mongodb","clickhouse"]
langs = ["python","typescript","elixir","go","rust","kotlin","swift","ruby"]
editors = ["vim","vscode","zed","emacs","cursor","helix","sublime","jetbrains"]
themes = ["dark mode","light mode","high-contrast mode","solarized"]
foods = ["vegetarian","vegan","gluten-free","pescatarian","no peanuts","halal","lactose-free"]
cities = ["singapore","madrid","berlin","lisbon","austin","tokyo","toronto","bangkok","london","seoul"]
days = ["monday","tuesday","wednesday","thursday","friday"]
times = ["9am","10:30am","1pm","3pm","4:30pm","6pm"]
tools = ["linear","jira","notion","github issues","trello","asana"]
clouds = ["aws","gcp","fly.io","hetzner","cloudflare","vercel","render"]
errs = ["timeout after 30s","out of memory","rate limit 429","deadlock on write","certificate expired","disk full","null pointer in parser"]
fmt = ["mp4","webm","mov","gif"]

T = [
 ("{p} prefers {e} with {t} for all coding sessions", "what editor setup does {p} like?", "pref"),
 ("{p} is {f}; avoid suggesting dishes with meat when planning team lunches", "any dietary restrictions for {p}?", "pref"),
 ("{p} is based in {c} and usually online from {tm} local time", "where does {p} live and when do they start work?", "fact"),
 ("the {pr} stores its data in {d}; do not propose migrating it", "which database backs {pr}?", "fact"),
 ("{pr} is written in {l} and deployed on {cl}", "what language and hosting does {pr} use?", "fact"),
 ("decided on {dy}: {pr} will keep {d} instead of switching, revisit next quarter", "did we agree to replace the datastore of {pr}?", "decision"),
 ("{p} owns {pr}; route review requests for it to them", "who should review changes to {pr}?", "fact"),
 ("weekly sync with {p} is every {dy} at {tm}", "when is the recurring meeting with {p}?", "fact"),
 ("bug in {pr}: {er} when the job runs longer than usual, fixed by adding a retry", "how did we resolve the failure in {pr}?", "episode"),
 ("{p} asked to always get a short summary first, then details", "how does {p} want answers formatted?", "pref"),
 ("tasks for {pr} are tracked in {to}, not in chat", "where do we track work items for {pr}?", "fact"),
 ("exports from {pr} must be {fm} at 1080p for the client", "what output format is required for {pr} deliverables?", "fact"),
 ("{p} hates being pinged after 9pm; batch non-urgent messages to the morning", "is it ok to message {p} late at night?", "pref"),
 ("last deploy of {pr} on {dy} rolled back because of {er}", "why was the recent release of {pr} reverted?", "episode"),
 ("{p} speaks {l2} natively and prefers docs in english", "which language should documentation for {p} be in?", "pref"),
 ("budget for {pr} hosting on {cl} is capped at ${n}/month", "how much can we spend running {pr}?", "fact"),
 ("the api key for {pr} rotates every {n2} days; remind {p} before it expires", "how often do credentials for {pr} change?", "fact"),
 ("{p} tested {pr} on {c} wifi and saw {n3}ms p95 latency", "what latency did {p} measure for {pr}?", "episode"),
]
l2s = ["italian","german","portuguese","japanese","spanish","hindi","korean","french"]

def fill(s, v): return s.format(**v)
def gen(n):
    out=[]
    for i in range(n):
        tpl, q, kind = T[i % len(T)] if i < len(T) else R.choice(T)
        v = dict(p=R.choice(people), e=R.choice(editors), t=R.choice(themes), f=R.choice(foods), c=R.choice(cities),
                 tm=R.choice(times), pr=R.choice(projects), d=R.choice(dbs), l=R.choice(langs), cl=R.choice(clouds),
                 dy=R.choice(days), er=R.choice(errs), to=R.choice(tools), fm=R.choice(fmt), l2=R.choice(l2s),
                 n=R.choice([20,50,100,250,500]), n2=R.choice([30,60,90]), n3=R.randint(40,900))
        out.append(dict(id=i+1, kind=kind, user_id=f"user-{R.randint(1,50)}", text=fill(tpl,v), query=fill(q,v),
                        created_at=1760000000 + i*37))
    return out
if __name__=="__main__":
    n=int(sys.argv[1]) if len(sys.argv)>1 else 10000
    json.dump(gen(n), open("memories.json","w"))
    print("wrote", n)
