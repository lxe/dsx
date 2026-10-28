"""Four independent coding requests, scored with fixed behavioral tests.

A small screening gate, not a substitute for a broad coding benchmark. Model
output is retained and executed in an isolated Python child with resource limits.
Run only on the disposable benchmark host. No external packages are required.
"""
import argparse, concurrent.futures, hashlib, json, os, pathlib, re, resource, statistics, subprocess, tempfile, time, urllib.request
TASKS = [
('intervals', '''Implement merge_intervals(intervals), accepting pairs (start,end) with start <= end. Return sorted disjoint tuples, merging overlaps and touching endpoints, without mutating input. Empty input returns [].''', '''
a=[(5,7),(1,3),(2,4),(7,8),(10,10)]
assert merge_intervals(a)==[(1,4),(5,8),(10,10)]
assert a==[(5,7),(1,3),(2,4),(7,8),(10,10)]
assert merge_intervals([])==[]
assert merge_intervals([(1,1),(1,1)])==[(1,1)]
assert merge_intervals([(-5,-1),(-1,0),(2,3)])==[(-5,0),(2,3)]
assert merge_intervals([(1,100),(2,3),(4,5)])==[(1,100)]
'''),
('toposort', '''Implement topological_sort(graph), where graph maps hashable node to an iterable of outgoing neighbors. Include nodes appearing only as neighbors, support disconnected graphs, don't mutate input, and raise ValueError for every directed cycle including a self-loop. Return a list; any valid topological order is accepted.''', '''
def check(g):
 order=topological_sort(g); nodes=set(g)
 for vs in g.values(): nodes.update(vs)
 assert len(order)==len(nodes) and set(order)==nodes
 pos={x:i for i,x in enumerate(order)}
 assert all(pos[u]<pos[v] for u,vs in g.items() for v in vs)
check({'a':['b','c'],'b':['d'],'z':[]})
check({}); check({1:[],2:[3],4:[3]})
for g in [{1:[1]},{1:[2],2:[1]},{1:[2],2:[3],3:[2]}, {1:[],2:[3],3:[2]}]:
 try: topological_sort(g)
 except ValueError: pass
 else: raise AssertionError('missed cycle')
'''),
('ttl_lru', '''Implement class TTLCache(capacity, ttl, clock) with methods put(key,value), get(key, default=None), and __len__. get/put are LRU accesses. Each put sets expiration to clock()+ttl; get does not refresh expiration. At expiration time the item is absent. Remove expired items before capacity eviction and when computing len. ttl can be 0, capacity 0 stores nothing, negative capacity raises ValueError. Use only Python standard library.''', '''
now=[0.0]; c=TTLCache(2,5,lambda:now[0])
c.put('a',1);c.put('b',2);assert c.get('a')==1
c.put('c',3);assert c.get('b','missing')=='missing'; assert len(c)==2
now[0]=4;assert c.get('a')==1
now[0]=5;assert c.get('a','x')=='x';assert len(c)==0
now[0]=10;c.put('a',1);now[0]=12;c.put('a',9)
now[0]=16;assert c.get('a')==9
now[0]=17;assert c.get('a') is None
z=TTLCache(0,10,lambda:0);z.put('x',1);assert len(z)==0
z=TTLCache(2,0,lambda:0);z.put('x',1);assert z.get('x') is None
try: TTLCache(-1,5,lambda:0)
except ValueError: pass
else: raise AssertionError('negative capacity accepted')
'''),
('jsonl', '''Implement class JSONLReader with feed(data: bytes)->list and finish()->list. Decode UTF-8 incrementally across arbitrary byte boundaries. Parse each nonblank newline-delimited JSON record, skipping malformed JSON records. finish parses a nonempty final record without a newline. Accept CRLF. Invalid UTF-8 may raise UnicodeDecodeError. Use Python standard library. No demonstration or tests in the output.''', r'''
import json
r=JSONLReader();b='{"x":"café😀"}\r\n\ninvalid\n[1,2]\nnull\n42'.encode()
out=[]
for v in b: out.extend(r.feed(bytes([v])))
out.extend(r.finish());assert out==[{'x':'café😀'},[1,2],None,42],out
r=JSONLReader();assert r.feed(b'')==[];assert r.finish()==[]
r=JSONLReader();assert r.feed(b'bad\nfalse\n{"a":')==[False]
assert r.feed(b'1}\n')==[{'a':1}];assert r.finish()==[]
''')]

def request(task, url):
 name,prompt,_=task
 body={'model':'deepseek-v4.1-flash','messages':[{'role':'system','content':'You are a coding assistant. Output only complete executable Python code. Do not include explanations.'},{'role':'user','content':prompt}], 'temperature':0,'seed':7100,'reasoning_effort':'none','max_tokens':3072,'stream':True,'stream_options':{'include_usage':True}}
 start=time.monotonic();first=None;content='';usage={};error=None;finish=None
 try:
  req=urllib.request.Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
  with urllib.request.urlopen(req,timeout=600) as response:
   for line in response:
    if not line.startswith(b'data: '):continue
    v=line[6:].strip()
    if v==b'[DONE]':break
    x=json.loads(v);usage=x.get('usage') or usage
    for c in x.get('choices',[]):
     finish=c.get('finish_reason') or finish
     t=c.get('delta',{}).get('content') or ''
     if t: first=first or time.monotonic();content+=t
 except Exception as e:error=str(e)
 end=time.monotonic();tok=usage.get('completion_tokens',0)
 return {'task':name,'text':content,'usage':usage,'error':error,'finish_reason':finish,'wall_s':end-start,'ttft_s':None if first is None else first-start,'decode_tps':None if first is None else (tok-1)/(end-first),'sha256':hashlib.sha256(content.encode()).hexdigest()}

def limits():
 resource.setrlimit(resource.RLIMIT_CPU,(5,5));resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2));resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2));resource.setrlimit(resource.RLIMIT_NOFILE,(32,32))

def score(result, tests):
 code=result['text'].strip()
 if code.startswith('```'):
  m=re.fullmatch(r'```(?:python|py)?\s*\n(.*?)\n```',code,re.S)
  if m:code=m.group(1)
 with tempfile.TemporaryDirectory(prefix='coding-gate-') as d:
  p=pathlib.Path(d)/'candidate.py';p.write_text(code+'\n'+tests+'\nprint("BEHAVIORAL_TESTS_PASSED")\n')
  try:
   r=subprocess.run(['/usr/bin/python3','-I',str(p)],cwd=d,env={'PATH':'/usr/bin:/bin'},capture_output=True,text=True,timeout=10,preexec_fn=limits)
   return {'passed':r.returncode==0 and 'BEHAVIORAL_TESTS_PASSED' in r.stdout,'returncode':r.returncode,'stdout':r.stdout[-2000:],'stderr':r.stderr[-4000:]}
  except Exception as e:return {'passed':False,'error':str(e)}

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--url',default='http://127.0.0.1:8090');ap.add_argument('--output',required=True);ap.add_argument('--label',required=True);a=ap.parse_args()
 started=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(lambda t:request(t,a.url),TASKS))
 wall=time.monotonic()-started
 for result,task in zip(results,TASKS):result['tests']=score(result,task[2]);result['label']=a.label
 out=pathlib.Path(a.output);out.write_text(''.join(json.dumps(r)+'\n' for r in results))
 print(json.dumps({'label':a.label,'passed':sum(r['tests']['passed'] for r in results),'tasks':4,'wall_s':wall,'decode_tps_median':statistics.median(r['decode_tps'] for r in results if r['decode_tps'] is not None),'requests':[{'task':r['task'],'tests':r['tests'],'decode_tps':r['decode_tps'],'finish_reason':r['finish_reason']} for r in results]}),flush=True)
