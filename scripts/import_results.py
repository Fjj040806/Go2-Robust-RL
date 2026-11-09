import csv, io, json, math, zipfile
from pathlib import Path
from statistics import mean

import argparse

parser = argparse.ArgumentParser(description='Import the original experiment audit and evaluation ZIPs')
parser.add_argument('archives', type=Path)
args = parser.parse_args()
dest = Path(__file__).resolve().parents[1] / 'data'
runs = {}
for p in sorted(args.archives.glob('pilot_*.zip')):
    with zipfile.ZipFile(p) as z:
        r = json.loads(z.read(next(n for n in z.namelist() if n.endswith('experiment.json'))))
        r.pop('run_dir_local')
        r.pop('evaluation_status')
        runs[r['checkpoint_sha256']] = r
assert len(runs) == 12
(dest / 'training_runs.json').write_text(json.dumps(sorted(runs.values(), key=lambda r:(r['group'],r['seed'])), indent=2)+'\n')

rows = []
seen = set()
for p in sorted(args.archives.glob('*.zip')):
    if not p.name.startswith(('results_', 'push_AD_')) or '200047' in p.name:
        continue
    with zipfile.ZipFile(p) as z:
        names = {n.replace('\\','/'):n for n in z.namelist()}
        for name,n in names.items():
            if not name.endswith('/summary.json'):
                continue
            report = json.loads(z.read(n))
            run = runs[report['policy_sha256']]
            group, seed = run['group'], run['seed']
            protocol = 'nominal' if report['protocol'].startswith('nominal') else 'timing' if 'timing' in report['protocol'] else 'fixed'
            assert (group,seed,protocol) not in seen
            seen.add((group,seed,protocol))
            assert report['status']=='completed' and report['eval_seed']==101
            for s in report['scenarios']:
                scenario = s['name']
                prefix = name.rsplit('/',1)[0]+'/'+scenario
                trace = z.read(names[prefix+'.csv']).decode('utf-8-sig').replace(chr(13) + chr(10), chr(10))
                samples = list(csv.DictReader(io.StringIO(trace)))
                fall = next((float(x['time_s']) for x in samples if x['terminated']=='True' and x['first_trajectory']=='True'),None)
                assert fall == s['first_fall_s']
                force = s.get('requested_force_world_y_N')
                onset = s.get('push_start_s',report.get('push_start_s')) if force else None
                recovery = None
                if force:
                    physics = list(csv.DictReader(io.StringIO(z.read(names[prefix+'_physics.csv']).decode())))
                    pulse = [x for x in physics if float(x['force_added_y_N'])!=0]
                    assert len(pulse)==20 and all(float(x['force_added_y_N'])==force for x in pulse)
                    assert math.isclose(float(pulse[0]['start_s']),onset) and math.isclose(float(pulse[-1]['end_s']),onset+.2)
                    if fall is None:
                        count = 0
                        for x in samples:
                            t = float(x['time_s'])
                            if t + 1e-9 < onset + .2:
                                continue
                            count = count+1 if abs(float(x['vy']))<=.2 else 0
                            if count==25:
                                recovery = round(t - 24*.02 - (onset+.2), 6)
                                break
                scored = [x for x in samples if float(x['time_s'])>2 and x['first_trajectory']=='True' and x['terminated']=='False']
                error = mean(math.hypot(float(x['vx'])-float(x['cmd_vx']),float(x['vy'])-float(x['cmd_vy'])) for x in scored) if scored else None
                if protocol=='nominal':
                    assert math.isclose(error,s['mean_planar_error_m_s'],abs_tol=1e-7)
                if group in ('A','D') and seed==7 and protocol=='fixed' and force==60:
                    (dest / f'example_{group}.csv').write_text(trace)
                rows.append(dict(group=group,train_seed=seed,protocol=protocol,scenario=scenario,cmd_vx=s['command'][0],cmd_vy=s['command'][1],cmd_wz=s['command'][2],force_N=force,onset_s=onset,fallen=int(fall is not None),fall_s=fall,recovery_latency_s=recovery,mean_planar_error_m_s=round(error,9) if error is not None else None))
assert len(seen)==36 and len(rows)==360
with (dest/'episodes.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]), lineterminator=chr(10))
    w.writeheader()
    w.writerows(sorted(rows,key=lambda r:(r['group'],r['train_seed'],r['protocol'],r['scenario'])))
