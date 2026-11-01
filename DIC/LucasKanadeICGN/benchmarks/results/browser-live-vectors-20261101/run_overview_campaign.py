import json,subprocess
from pathlib import Path
root=Path('/home/jeff/Code/tests_gpuio')
output=Path('/tmp/live-vector-overview-bench');output.mkdir(exist_ok=True)
assert Path('/tmp/live-vectors-overview-tests.log').read_text().strip().endswith('OK')
commands=[]
for index,variant in enumerate(['baseline','candidate','candidate','baseline'],1):
 site=Path('/tmp/live-dic-fixed-bench-candidate-site' if variant=='baseline' else '/tmp/live-dic-live-vectors-overview-final-site')
 name=f'{index}-{variant}'
 argv=[str(root/'DIC/LucasKanadeICGN/python/.venv/bin/python'),str(root/'tools/diagnose_camera_pipeline.py'),'--site',str(site),'--output',str(output/(name+'.json')),'--resolution','3840x2160','--duration','20','--warmup','4','--comparison','fixed','--recording','all','--solver-profile','common-zero','--storage-probe','--overlap-pipeline','--display-rate','10','--preview-width','1920']
 commands.append(dict(name=name,argv=argv));(output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
 print('Starting',name,flush=True)
 with (output/(name+'.stdout')).open('w') as stdout,(output/(name+'.stderr')).open('w') as stderr:subprocess.run(argv,stdout=stdout,stderr=stderr,check=True)
 d=json.loads((output/(name+'.json')).read_text());print(name,[(c['recording'],round(c['completed_fps'],3),round(c['committed_fps'],3)) for c in d['cases']],flush=True)
print('COMPLETE',flush=True)
