"""Publication-style static figures from actual tracking/evaluation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_tracks(intake):
    paths=sorted((intake/'tracks').glob('*.raw_tracks.npz'))
    fig,axes=plt.subplots(4,4,figsize=(15,11),sharex=True,sharey=True)
    for ax,path in zip(axes.ravel(),paths):
        with np.load(path,allow_pickle=False) as z:
            obj=z['object_xy']*100;tool=z['tool_xy']*100
        ax.plot(tool[:,0],tool[:,1],color='#2476b8',lw=1.2,alpha=.7,label='Tool')
        ax.plot(obj[:,0],obj[:,1],color='#d24437',lw=1.8,label='Object')
        valid=np.flatnonzero(np.isfinite(obj).all(axis=1))
        if len(valid):
            ax.scatter(*obj[valid[0]],color='#d24437',marker='o',s=24,zorder=3)
            ax.scatter(*obj[valid[-1]],color='#d24437',marker='s',s=24,zorder=3)
        ax.set_title(path.name.split('.')[0].replace('_',' ').title(),fontsize=11)
        ax.set_xlim(-20,20);ax.set_ylim(-15,15);ax.set_aspect('equal')
        ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Table x (cm)')
    for ax in axes[:,0]:ax.set_ylabel('Table y (cm)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper right',ncol=2,frameon=False)
    fig.suptitle('Applicant-recorded trajectories after per-frame camera correction',fontsize=16,y=.99)
    fig.text(.5,.013,'40 x 30 cm measured table | circles: object starts | squares: object ends | gaps remain visible',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.03,1,.96))
    fig.savefig(intake/'recorded_trajectories.png',dpi=160)
    fig.savefig(intake/'recorded_trajectories.svg')
    plt.close(fig)


def plot_results(run):
    prediction=json.loads((run/'prediction_metrics.json').read_text())
    metrics=prediction['learned']
    names=['Learned','Shuffled actions','Stationary','Constant velocity','Copy tool']
    values=[metrics['model_moving_mae_mm'],prediction['action_shuffled']['model_moving_mae_mm'],
            metrics['stationary_moving_mae_mm'],metrics['constant_velocity_moving_mae_mm'],metrics['copy_tool_moving_mae_mm']]
    fig,ax=plt.subplots(figsize=(9,4.4))
    colors=['#2476b8','#9aafbf','#b7bcc1','#b7bcc1','#b7bcc1']
    bars=ax.bar(names,values,color=colors)
    for bar,value in zip(bars,values):
        if value is not None:ax.text(bar.get_x()+bar.get_width()/2,value,f'{value:.2f}',ha='center',va='bottom')
    ax.set_ylim(0,max(x for x in values if x is not None)*1.2)
    ax.set_ylabel('Mean displacement error (mm; lower is better)')
    ax.set_title(f"Held-out recorded data: moving transitions only (n={metrics['moving_transitions']})")
    ax.spines[['top','right']].set_visible(False)
    fig.tight_layout();fig.savefig(run/'prediction_comparison.png',dpi=160);plt.close(fig)
    summary=json.loads((run/'summary.json').read_text())
    conditions=['nominal','impulse','friction'];modes=['replay','geometry','learned']
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    for ax,condition in zip(axes,conditions):
        rows=[next(r for r in summary if r['condition']==condition and r['mode']==m) for m in modes]
        for x,(mode,row,color) in enumerate(zip(modes,rows,['#aeb5bd','#e8aa48','#2476b8'])):
            y=row['success_rate']
            ax.bar(x,y,color=color)
            ax.errorbar(x,y,yerr=[[y-row['wilson95_low']],[row['wilson95_high']-y]],color='#253746',capsize=4)
            ax.text(x,min(y+.08,1.12),f"{row['successes']}/{row['n']}",ha='center')
        ax.set_xticks(range(3),['Replay','Geometry','Learned']);ax.set_title(condition.title());ax.set_ylim(0,1.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Simulation success fraction')
    fig.suptitle('Same held-out references and reset seeds for every controller',fontsize=14)
    fig.text(.5,.015,'Whiskers: descriptive Wilson intervals; rollouts sharing a reference are correlated.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.92));fig.savefig(run/'control_comparison.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--intake',type=Path)
    parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    if args.intake:plot_tracks(args.intake)
    if args.run:plot_results(args.run)

