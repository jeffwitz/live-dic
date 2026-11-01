"""Run the predeclared fixed-reference qualification sequentially."""
import json, os, subprocess, sys
from pathlib import Path

root=Path.cwd()
python=root/'DIC/LucasKanadeICGN/python/.venv/bin/python'
output=Path('/tmp/live-dic-fixed-reference-bench');output.mkdir(exist_ok=True)
baseline=Path('/tmp/live-dic-fixed-bench-baseline')
assert Path('/tmp/live-dic-fixed-bench-tests.log').read_text().strip().endswith('OK'), 'Finish tests before timing'
commands=[]
def run(name,args,source=root):
    env=os.environ.copy();env['PYTHONPATH']=str(source/'DIC/LucasKanadeICGN/python/src')
    argv=[str(python),*map(str,args)]
    commands.append({'name':name,'argv':argv,'pythonpath':env['PYTHONPATH']})
    (output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    print('Starting',name,flush=True)
    with (output/(name+'.stdout')).open('w') as stdout,(output/(name+'.stderr')).open('w') as stderr:
        subprocess.run(argv,env=env,stdout=stdout,stderr=stderr,check=True)
    if name.startswith('python-') or name.startswith('native-depth2'):
        data=json.loads((output/(name+'.stdout')).read_text());(output/(name+'.json')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
        print(name,{k:{m:round(v[m],3) for m in ['latency_median_ms','latency_p95_ms','throughput_fps']} for k,v in data['measurements'].items()},flush=True)
    else:
        data=json.loads((output/(name+'.json')).read_text())
        print(name,[(c['recording'],round(c['completed_fps'],3),round(c['committed_fps'],3)) for c in data['cases']],flush=True)

benchmark=root/'DIC/LucasKanadeICGN/benchmarks/benchmark_image_to_field.py'
common=['--fixture-set','uhd-tensile','--width','3840','--height','2160','--frames','300','--warmups','20','--campaigns','1','--pyramid-levels','2','--subset-size','16','--subset-stride','16','--interpolation','bspline','--motion-model','affine','--outlier-policy','median']
for backend in ['native-pyramid','webgpu-wgsl']:
    for index,version in enumerate(['baseline','baseline','candidate','baseline','candidate','baseline'],1):
        extra=['--device-type','gpu'] if backend=='native-pyramid' else []
        run(f'python-{backend}-{index}-{version}',[benchmark,*common,'--pipeline-depth','1','--backend',backend,*extra],baseline if version=='baseline' else root)
for index in [1,2]:run(f'native-depth2-{index}',[benchmark,*common,'--pipeline-depth','2','--backend','native-pyramid','--device-type','gpu'])

for index,version in enumerate(['baseline','baseline','candidate-default','candidate-preview','baseline','candidate-preview','candidate-default','baseline'],1):
    site=Path('/tmp/live-dic-fixed-bench-baseline-site') if version=='baseline' else Path('/tmp/live-dic-fixed-bench-candidate-site')
    name=f'pwa-{index}-{version}'
    extra=['--overlap-pipeline','--display-rate','1','--preview-width','1920'] if version=='candidate-preview' else []
    run(name,[root/'tools/diagnose_camera_pipeline.py','--site',site,'--output',output/(name+'.json'),'--resolution','3840x2160','--duration','20','--warmup','4','--comparison','fixed','--recording','all','--solver-profile','common-zero','--storage-probe',*extra])
print('COMPLETE',flush=True)
