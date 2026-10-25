"""Recreate analysis proxies from the recorded intake manifests; preserve originals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg

from phone2sim.data import sha256


def prepare(root, overwrite=False):
    root = Path(root)
    intake = json.loads((root / 'intake_manifest.json').read_text(encoding='utf-8'))
    proxies = json.loads((root / 'proxy_manifest.json').read_text(encoding='utf-8'))
    originals = {row['sha256']: row for row in intake['files']}
    for row in proxies:
        source, target = Path(row['source']), Path(row['proxy'])
        if not target.resolve().is_relative_to(root.resolve()):
            raise ValueError('Proxy output must stay inside the intake directory.')
        source_hash = sha256(source)
        if source_hash not in originals:
            raise ValueError(f'Original recording hash is not in the intake: {source.name}')
        calibration = root / 'tracks' / f"clip_{row['clip']:02d}.calibration.json"
        if target.exists() and not overwrite:
            if not calibration.exists():
                raise ValueError('Existing proxy has no saved hash; inspect it before --overwrite.')
            expected = json.loads(calibration.read_text())['proxy_sha256']
            if sha256(target) != expected:
                raise ValueError(f'Existing proxy hash changed: {target.name}')
            print(f'Verified existing {target.name}', flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            imageio_ffmpeg.get_ffmpeg_exe(), '-nostdin', '-hide_banner', '-loglevel', 'error',
            '-y', '-i', str(source), '-an', '-vf', ','.join(row['filters']),
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p', str(target),
        ], check=True)
        print(f'Created {target.name}; reprocess tracking to record its new hash.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intake', required=True)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    prepare(args.intake, args.overwrite)
