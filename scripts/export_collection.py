"""Export reviewed metric tracks, preserving which observations were genuinely detected."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from phone2sim.data import Episode,sha256,write_json
from phone2sim.vision import fill_gaps


def export(root,output,reviewed):
    if not reviewed:
        raise ValueError('Inspect the QA artifacts and explicitly record --reviewed-tracks before export.')
    root,output=Path(root),Path(output)
    if output.exists():
        raise ValueError('Use a new processed output directory; existing datasets are preserved.')
    paths=sorted((root/'tracks').glob('*.raw_tracks.npz'))
    loaded=[]
    diameters=[]
    for path in paths:
        with np.load(path,allow_pickle=False) as z:
            row={k:z[k].copy() for k in ('time','object_xy','tool_xy','silhouette_diameters_m')}
            row['metadata']=json.loads(str(z['metadata']))
        # Static initial image observations estimate hardware size, not future outcomes.
        diameters.extend(row['silhouette_diameters_m'][:5,0][np.isfinite(row['silhouette_diameters_m'][:5,0])].tolist())
        loaded.append((path,row))
    if not diameters:
        raise ValueError('No object silhouette measurements available.')
    object_diameter=float(np.round(np.median(diameters),3))
    output.mkdir(parents=True)
    included,excluded=[],[]
    for path,row in loaded:
        metadata=row['metadata']
        obj,tool=row['object_xy'],row['tool_xy']
        observed=np.isfinite(obj).all(axis=1)&np.isfinite(tool).all(axis=1)
        if not observed.any():
            excluded.append({'episode':metadata['episode_id'],'reason':'No jointly observed frames.'})
            continue
        first,last=np.flatnonzero(observed)[[0,-1]]
        section=slice(first,last+1)
        obj,tool=obj[section],tool[section]
        try:
            obj,n_obj=fill_gaps(obj,max_gap=2,max_missing=.10)
            tool,n_tool=fill_gaps(tool,max_gap=2,max_missing=.10)
            jumps=max(np.linalg.norm(np.diff(x,axis=0),axis=1).max() for x in (obj,tool))
            if jumps>.04:
                raise ValueError('Track jump >40 mm/sample; inspect rather than smoothing it away.')
            metadata.update({
                'object_diameter_m':object_diameter,
                'object_diameter_source':'median calibrated top-silhouette estimate in initial five samples per recording; rounded to 1 mm; not a measured contact diameter',
                'object_diameter_initial_estimates_min_m':float(np.min(diameters)),
                'object_diameter_initial_estimates_max_m':float(np.max(diameters)),
                'tracking_reviewed':True,
                'tracking_review_scope':'visual QA of sampled overlays plus all-sample geometry/continuity checks',
                'trim_start_seconds':float(row['time'][first]),'trim_end_seconds':float(row['time'][last]),
                'interpolated_object_samples':n_obj,'interpolated_tool_samples':n_tool,
                'interpolation_policy':'at most 2 samples per gap; all transitions touching fills are excluded from training/prediction scoring',
                'raw_tracks_sha256':sha256(path),
                'tracking_implementation_sha256':sha256(Path('phone2sim/planar_tracking.py')),
                'processing_script_sha256':sha256(Path('scripts/process_collection.py')),
                'export_script_sha256':sha256(Path(__file__)),
            })
            episode=Episode(row['time'][section]-row['time'][first],obj,tool,metadata,observed[section])
            target=output/f"{metadata['episode_id']}.npz"
            episode.save(target)
            included.append({'episode':metadata['episode_id'],'file':target.name,'samples':len(episode.time),
                             'interpolated_samples':int((~episode.valid_observations).sum()),
                             'max_step_mm':float(jumps*1000)})
        except ValueError as error:
            excluded.append({'episode':metadata['episode_id'],'reason':str(error)})
    summary={'origin':'human_video','source_collection':'user-confirmed independent recordings',
             'board_width_m':.4,'board_height_m':.3,'tool_diameter_m':.0235,
             'object_diameter_m_image_estimate':object_diameter,'included':included,'excluded':excluded,
             'note':'No synthetic transitions were added. Missing observations are flagged and excluded from prediction learning/scoring.'}
    write_json(output/'dataset_summary.json',summary)
    write_json(root/'dataset_export.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--intake',required=True)
    parser.add_argument('--out',required=True)
    parser.add_argument('--reviewed-tracks',action='store_true')
    args=parser.parse_args()
    export(args.intake,args.out,args.reviewed_tracks)

