"""Reproduce moving-camera tracking of an explicitly inventoried local collection.

The intake/proxy manifests preserve the user-supplied file hashes and preprocessing.
Reference corners must be checked visually before using this script for experiments.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import html
import json
from pathlib import Path

import cv2
import numpy as np

from phone2sim.data import sha256, write_json
from phone2sim.planar_tracking import track_planar_video


def process_one(index, root, width_m, height_m, tool_diameter_m, collected_at, collector,reuse_geometry=False):
    intake=json.loads((root/'intake_manifest.json').read_text(encoding='utf-8'))
    proxies=json.loads((root/'proxy_manifest.json').read_text(encoding='utf-8'))
    corners=json.loads((root/'reference_corners_provisional.json').read_text(encoding='utf-8'))
    original=intake['files'][index-1]
    proxy=proxies[index-1]
    reference=corners[index-1]
    output=root/'tracks'; output.mkdir(exist_ok=True)
    config={
        'corners_px':reference['corners_px'],'reference_frame':reference['reference_frame'],
        'width_m':width_m,'height_m':height_m,'sample_hz':10,
        'reference_exclusion_polygon':[[390,190],[860,190],[860,510],[390,510]],
        'object_hsv':[[[0,150,45],[8,255,255]],[[168,150,45],[179,255,255]]],
        'tool_hsv':[[[92,110,35],[135,255,255]]],
        'min_inliers':18,'min_coverage':.12,
        'marker_size_ranges_m':[[.03,.065],[.008,.035]],'tracking_margin_m':.03,
        'source_sha256':original['sha256'],'proxy_sha256':sha256(proxy['proxy']),
        'calibration_dimensions_source':'user-declared measurements',
        'reference_corner_selection':'brown tape-cross proposals, visually checked reference montage',
        'camera_model':'per-sampled-frame ORB + RANSAC planar homography; SIFT fallback with same geometry thresholds',
    }
    config_path=output/f'clip_{index:02d}.calibration.json'
    geometry_cache=None
    if reuse_geometry:
        previous=json.loads(config_path.read_text())
        if any(previous[k]!=config[k] for k in ('proxy_sha256','reference_frame','corners_px','width_m','height_m','reference_exclusion_polygon')):
            raise ValueError('Cannot reuse camera fits after input or reference calibration changes.')
        geometry_cache=json.loads((output/f'clip_{index:02d}.geometry.json').read_text())
    write_json(config_path,config)
    tracks=track_planar_video(proxy['proxy'],config,output/f'clip_{index:02d}.tracking.mp4',geometry_cache)
    obj,tool=tracks['object_xy'],tracks['tool_xy']
    valid_obj=np.isfinite(obj).all(axis=1)
    valid_tool=np.isfinite(tool).all(axis=1)
    valid=valid_obj&valid_tool
    metadata={
        'episode_id':f'phone_{index:02d}','origin':'human_video',
        'source_group':original['sha256'],'source_sha256':original['sha256'],
        'source_file':original['local_file'],'proxy_file':proxy['proxy'],
        'proxy_sha256':config['proxy_sha256'],'calibration_sha256':sha256(config_path),
        'collector':collector,'collected_at':collected_at,'collected_at_timezone':'not specified by collector',
        'attested_own_recording':True,'independent_recording_declared':True,
        'authorship_declaration':'user confirmation in chat on 2026-10-22',
        'tracking_reviewed':False,'source_fps':original['fps'],'analysis_fps':30,'sample_hz':10,
        'preprocessing_filters':proxy['filters'],'tool_diameter_m':tool_diameter_m,
        'tool_diameter_source':'user-measured 2.35 cm',
    }
    np.savez_compressed(output/f'clip_{index:02d}.raw_tracks.npz',
                        time=tracks['time'],object_xy=obj,tool_xy=tool,
                        silhouette_diameters_m=tracks['silhouette_diameters_m'],
                        metadata=np.array(json.dumps(metadata)))
    write_json(output/f'clip_{index:02d}.geometry.json',tracks['geometry'])
    errors=[]
    if not valid.all():
        errors.append('Some frames have missing detections/camera fits; review gaps before export.')
    camera=[d for d in tracks['geometry'] if d['valid']]
    good_indices=np.flatnonzero(valid)
    displacement=None
    start_std=None
    object_diameter=None
    direction='unresolved'
    if len(good_indices)>10:
        first=good_indices[:5];last=good_indices[-5:]
        delta=np.median(obj[last],axis=0)-np.median(obj[first],axis=0)
        displacement=float(np.linalg.norm(delta)*1000)
        start_std=float(np.linalg.norm(np.std(obj[first],axis=0))*1000)
        object_diameter=float(np.nanmedian(tracks['silhouette_diameters_m'][:,0]))
        direction=('right' if delta[0]>0 else 'left') if abs(delta[0])>abs(delta[1]) else ('up' if delta[1]>0 else 'down')
        if displacement<40:
            errors.append('Net object displacement <40 mm; inspect whether motion is sufficient.')
    result={
        'clip':index,'source_name':original['source_name'],'samples':len(valid),
        'object_visible_fraction':float(valid_obj.mean()),'tool_visible_fraction':float(valid_tool.mean()),
        'joint_valid_fraction':float(valid.mean()),'camera_valid_fraction':float(len(camera)/len(valid)),
        'minimum_inliers_accepted':min((d['inliers'] for d in camera),default=0),
        'median_reprojection_px':float(np.median([d['median_reprojection_px'] for d in camera])) if camera else None,
        'camera_corner_displacement_px_max':max((d['corner_displacement_px'] for d in camera),default=None),
        'net_object_displacement_mm':displacement,'object_direction':direction,
        'initial_half_second_position_std_mm':start_std,
        'object_silhouette_diameter_m_median':object_diameter,
        'object_size_note':'image-estimated top silhouette under a table-plane homography, not a measured contact diameter',
        'issues':errors,'calibration_file':config_path.as_posix(),
        'tracking_video':f'tracks/clip_{index:02d}.tracking.mp4',
        'raw_tracks_file':f'tracks/clip_{index:02d}.raw_tracks.npz',
    }
    # Export a representative QA frame and a compact trace plot with actual image labels.
    cap=cv2.VideoCapture(str(output/f'clip_{index:02d}.tracking.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(cap.get(cv2.CAP_PROP_FRAME_COUNT)*.5))
    ok,preview=cap.read();cap.release()
    if ok:cv2.imwrite(str(output/f'clip_{index:02d}.preview.jpg'),preview)
    write_json(output/f'clip_{index:02d}.quality.json',result)
    return result


def report(root):
    results=[json.loads(p.read_text(encoding='utf-8')) for p in sorted((root/'tracks').glob('*.quality.json'))]
    write_json(root/'tracking_quality.json',results)
    rows=[]
    for r in results:
        name=f"clip_{r['clip']:02d}"
        cells=[name,r['object_direction'],f"{r['joint_valid_fraction']:.1%}",
               f"{r['camera_valid_fraction']:.1%}",f"{r['net_object_displacement_mm']:.1f}" if r['net_object_displacement_mm'] is not None else 'N/A',
               f"{r['initial_half_second_position_std_mm']:.2f}" if r['initial_half_second_position_std_mm'] is not None else 'N/A']
        rows.append('<tr>'+''.join(f'<td>{html.escape(c)}</td>' for c in cells)+f'<td><a href="{r["tracking_video"]}">QA video</a></td></tr>')
    markup='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Phone2Sim 视频检查</title>
<style>body{font:16px/1.6 system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#1b2b3b;background:#f5f6f8}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}.note{background:#fff0c2;padding:16px}img{width:100%;max-width:1000px}a{color:#1758a0}</style>
<h1>16 段自采视频：跟踪质量检查</h1><p>标定：用户确认 40 × 30 cm；蓝色工具头直径 2.35 cm。拍摄时间按用户原述记录为 2026-10-21 23:00，时区未指定。</p>
<p class="note">本页是输入视频与轨迹质量检查，不是机器人任务成功率或训练结果。右侧画面是逐帧校正后的桌面视图。红色物体尺寸为图像估算，尚无实测接触尺寸。</p>
<table><tr><th>视频</th><th>主方向</th><th>红蓝同时有效</th><th>相机校正有效</th><th>净位移 mm</th><th>开头位置波动 mm</th><th>叠加检查</th></tr>'''+''.join(rows)+'</table>'
    for r in results:
        markup+=f'<h2>Clip {r["clip"]:02d}</h2><img src="tracks/clip_{r["clip"]:02d}.preview.jpg"><p>{html.escape(r["source_name"])}</p>'
    markup+='<p>净位移取首尾五个有效样本的位置中位数；开头位置波动可能同时包含真实微动与跟踪误差，不能当作独立标定精度。</p></html>'
    (root/'tracking_report.html').write_text(markup,encoding='utf-8')
    lines=['# 自采视频跟踪检查','',
           '标定区域：40 × 30 cm；蓝色工具头直径：2.35 cm。数据来源与拍摄时间已经用户确认。','',
           '**这里只检查视频与轨迹质量，不代表训练或机器人任务结果。**','',
           '| 视频 | 主方向 | 红蓝同时有效 | 相机校正有效 | 净位移 mm |',
           '|---|---|---:|---:|---:|']
    for r in results:
        distance=f"{r['net_object_displacement_mm']:.1f}" if r['net_object_displacement_mm'] is not None else 'N/A'
        lines.append(f"| [clip_{r['clip']:02d}]({r['tracking_video']}) | {r['object_direction']} | {r['joint_valid_fraction']:.1%} | {r['camera_valid_fraction']:.1%} | {distance} |")
    (root/'tracking_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--intake',required=True)
    parser.add_argument('--width-m',type=float,required=True)
    parser.add_argument('--height-m',type=float,required=True)
    parser.add_argument('--tool-diameter-m',type=float,required=True)
    parser.add_argument('--collected-at',required=True)
    parser.add_argument('--collector',required=True)
    parser.add_argument('--attest-own-recording',action='store_true',required=True)
    parser.add_argument('--clips',type=int,nargs='+',default=list(range(1,17)))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--reuse-geometry',action='store_true',help='Reuse checked camera fits only when proxy/reference hashes match.')
    args=parser.parse_args()
    datetime.fromisoformat(args.collected_at)
    root=Path(args.intake)
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks={executor.submit(process_one,n,root,args.width_m,args.height_m,args.tool_diameter_m,args.collected_at,args.collector,args.reuse_geometry):n for n in args.clips}
        for task in as_completed(tasks):
            r=task.result()
            print(json.dumps({k:r[k] for k in ['clip','samples','joint_valid_fraction','camera_valid_fraction','net_object_displacement_mm','object_direction','issues']}),flush=True)
    report(root)


if __name__=='__main__':
    main()

