import json, os, subprocess
from pathlib import Path

root=Path.cwd();out=Path('/tmp/live-dic-fixed-reference-bench')
assert Path('/tmp/live-dic-fixed-reference-bench.log').read_text().rstrip().endswith('COMPLETE')
commands=json.loads((out/'commands.json').read_text())
template=next(row for row in commands if row['name']=='python-webgpu-wgsl-1-baseline')['argv']
extra=[]
for index,version in enumerate(['baseline','candidate','candidate','baseline'],1):
 name=f'followup-webgpu-{index}-{version}'
 source=Path('/tmp/live-dic-fixed-bench-baseline') if version=='baseline' else root
 env=os.environ.copy();env['PYTHONPATH']=str(source/'DIC/LucasKanadeICGN/python/src')
 extra.append({'name':name,'argv':template,'pythonpath':env['PYTHONPATH']});(out/'followup-commands.json').write_text(json.dumps(extra,indent=2)+'\n')
 with (out/(name+'.stdout')).open('w') as stdout,(out/(name+'.stderr')).open('w') as stderr:subprocess.run(template,env=env,stdout=stdout,stderr=stderr,check=True)
 report=json.loads((out/(name+'.stdout')).read_text());(out/(name+'.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 case=report['measurements']['webgpu-wgsl'];print(name,{key:case[key] for key in ['throughput_fps','latency_median_ms','latency_p95_ms']},flush=True)
