"""Calibrated colour-marker tracking; intentionally no opaque vision service."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

import cv2
import numpy as np
import imageio.v2 as imageio

from .data import Episode, sha256, write_json


def homography(corners, width_m, height_m):
    corners = np.asarray(corners, np.float32)
    if corners.shape != (4, 2) or min(width_m, height_m) <= 0:
        raise ValueError("Provide four ordered corners and positive measured dimensions.")
    if not cv2.isContourConvex(corners.astype(np.float32)) or abs(cv2.contourArea(corners)) < 100:
        raise ValueError("Corners must form a non-degenerate convex quadrilateral in cyclic order.")
    # Image top-left -> world (-W/2,+H/2), so world y points up.
    target = np.array([[-width_m/2, height_m/2], [width_m/2, height_m/2],
                       [width_m/2, -height_m/2], [-width_m/2, -height_m/2]], np.float32)
    return cv2.getPerspectiveTransform(corners, target)


def calibrate(video, output, width_m=.4, height_m=.3):
    cap = cv2.VideoCapture(str(video))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Cannot decode first video frame: {video}")
    points = []
    label = "Click TL, TR, BR, BL of measured mat; Enter saves; R resets; Esc cancels"
    def click(event, x, y, flags, parameter):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Calibration", click)
    try:
        while True:
            canvas = frame.copy()
            cv2.putText(canvas, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
            for i, p in enumerate(points):
                cv2.circle(canvas, tuple(p), 5, (0, 255, 255), -1)
                cv2.putText(canvas, str(i+1), tuple(p), 0, .7, (0, 0, 0), 2)
            cv2.imshow("Calibration", canvas)
            key = cv2.waitKey(30) & 0xff
            if key == 27 or cv2.getWindowProperty("Calibration", cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Calibration cancelled.")
            if key in (ord("r"), ord("R")):
                points.clear()
            if key == 13 and len(points) == 4:
                homography(points, width_m, height_m)
                break
    finally:
        cv2.destroyAllWindows()
    config = {"corners_px": points, "width_m": width_m, "height_m": height_m,
              "frame_size": [frame.shape[1], frame.shape[0]], "sample_hz": 10,
              "object_hsv": [[[0, 100, 65], [12, 255, 255]], [[168, 100, 65], [179, 255, 255]]],
              "tool_hsv": [[[92, 100, 65], [130, 255, 255]]],
              "min_blob_px": 25, "max_gap_samples": 2, "max_missing_fraction": .10}
    write_json(output, config)
    return config


def centroid(frame, ranges, polygon=None, min_pixels=25):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(frame.shape[:2], np.uint8)
    for lower, upper in ranges:
        mask |= cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    if polygon is not None:
        roi = np.zeros_like(mask)
        cv2.fillConvexPoly(roi, np.asarray(polygon, np.int32), 255)
        mask &= roi
    count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
    if count < 2:
        return None
    sizes = stats[1:, cv2.CC_STAT_AREA]
    best = int(np.argmax(sizes))+1
    return centers[best] if stats[best, cv2.CC_STAT_AREA] >= min_pixels else None


def fill_gaps(track, max_gap=2, max_missing=.10):
    track = np.asarray(track, dtype=float)
    valid = np.isfinite(track).all(axis=1)
    if len(track) < 12 or valid.mean() < 1-max_missing:
        raise ValueError("Too many missing detections; fix lighting/HSV ranges or re-record.")
    if not valid[0] or not valid[-1]:
        raise ValueError("First and last samples need visible object and tool markers; trim the video.")
    longest = run = 0
    for item in valid:
        run = 0 if item else run+1
        longest = max(longest, run)
    if longest > max_gap:
        raise ValueError(f"Occlusion lasts {longest} samples; limit is {max_gap}. Re-record or annotate.")
    filled = np.column_stack([np.interp(np.arange(len(track)), np.flatnonzero(valid), track[valid, i]) for i in range(2)])
    return filled, int((~valid).sum())


def extract_tracks(video, calibration):
    """Decode tracks without declaring their authorship; also usable on test fixtures."""
    video, calibration = Path(video), Path(calibration)
    config = json.loads(calibration.read_text(encoding="utf-8-sig"))
    matrix = homography(config["corners_px"], config["width_m"], config["height_m"])
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hz = float(config.get("sample_hz", 10))
    if not cap.isOpened() or not 0 < hz <= fps <= 240:
        cap.release()
        raise ValueError("Cannot decode video or sample rate exceeds video FPS (supported <=240).")
    object_points, tool_points, times = [], [], []
    previews = []
    index, next_time = 0, 0.
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if [frame.shape[1], frame.shape[0]] != config["frame_size"]:
                raise ValueError("Calibration frame size does not match video. Recalibrate this recording.")
            current = index/fps
            index += 1
            if current+1e-6 < next_time:
                continue
            row = []
            for key, colour in [("object_hsv", (0, 0, 255)), ("tool_hsv", (255, 130, 0))]:
                center = centroid(frame, config[key], config["corners_px"], config.get("min_blob_px", 25))
                if center is None:
                    row.append([np.nan, np.nan])
                else:
                    projected = cv2.perspectiveTransform(np.float32([[center]]), matrix)[0, 0]
                    row.append(projected)
                    cv2.circle(frame, tuple(center.astype(int)), 7, colour, 2)
            object_points.append(row[0]); tool_points.append(row[1]); times.append(current)
            cv2.putText(frame, "TRACKING QA - inspect both markers before training", (10, 24), 0, .5, (255, 255, 255), 1)
            previews.append(cv2.cvtColor(cv2.resize(frame, (640, int(frame.shape[0]*640/frame.shape[1]))), cv2.COLOR_BGR2RGB))
            next_time += 1/hz
    finally:
        cap.release()
    detected=np.isfinite(np.asarray(object_points)).all(axis=1)&np.isfinite(np.asarray(tool_points)).all(axis=1)
    obj, missing_obj = fill_gaps(object_points, config.get("max_gap_samples", 2), config.get("max_missing_fraction", .1))
    tool, missing_tool = fill_gaps(tool_points, config.get("max_gap_samples", 2), config.get("max_missing_fraction", .1))
    grid = np.arange(int(np.floor(times[-1]*hz))+1)/hz
    obj = np.column_stack([np.interp(grid, times, obj[:, i]) for i in range(2)])
    tool = np.column_stack([np.interp(grid, times, tool[:, i]) for i in range(2)])
    # No temporal smoothing: preserve causal actions. Flag obvious tracking jumps.
    if max(np.linalg.norm(np.diff(x, axis=0), axis=1).max() for x in (obj, tool)) > .06:
        raise ValueError("Track jumps >6 cm/sample; inspect marker identity, calibration and motion speed.")
    stats = {"sample_hz": hz, "source_fps": fps,
             "interpolated_object_samples": missing_obj, "interpolated_tool_samples": missing_tool}
    right=np.clip(np.searchsorted(times,grid,side='left'),0,len(times)-1)
    left=np.maximum(right-1,0)
    exact=np.isclose(np.asarray(times)[right],grid,atol=1e-7)
    observed=detected[right]&(exact|detected[left])
    stats['observation_valid']=observed.tolist()
    return grid, obj, tool, stats, previews


def track_video(video, calibration, output, collector, collected_at, own_recording,
                source_group=None, episode_id=None):
    if not own_recording or not collector.strip():
        raise ValueError("Supply your collector name and --attest-own-recording for a video you personally recorded.")
    datetime.fromisoformat(collected_at)
    video, output, calibration = Path(video), Path(output), Path(calibration)
    if output.exists():
        raise ValueError("Output episode already exists; choose a new output name.")
    grid, obj, tool, stats, previews = extract_tracks(video, calibration)
    observation_valid=np.asarray(stats.pop('observation_valid'),dtype=bool)
    metadata = {"episode_id": episode_id or output.stem, "origin": "human_video",
                "collector": collector.strip(), "collected_at": collected_at,
                "source_sha256": sha256(video), "calibration_sha256": sha256(calibration),
                "source_group": source_group or sha256(video), "attested_own_recording": True,
                **stats,
                "tracking_reviewed": False}
    episode = Episode(grid, obj, tool, metadata,observation_valid)
    episode.save(output)
    qa_path = output.with_suffix(".tracking.mp4")
    imageio.mimsave(qa_path, previews, fps=stats["sample_hz"], macro_block_size=2)
    write_json(output.with_suffix(".provenance.json"), metadata)
    return episode

