"""Track coloured tools on a moving-camera table using static background features.

The homography is fitted independently to each sampled frame. Failed geometry
is missing data, never silently replaced with a stale camera transform. The
reference exclusion polygon must cover movable objects/hands in the reference.
"""
from __future__ import annotations

import cv2
import numpy as np


class PlanarTracker:
    def __init__(self, reference, corners_px, width_m, height_m, exclusion_polygon=None,
                 min_inliers=18, min_coverage=.12):
        self.corners = np.asarray(corners_px, np.float32)
        self.width_m, self.height_m = width_m, height_m
        target = np.float32([[-width_m/2, height_m/2], [width_m/2, height_m/2],
                             [width_m/2, -height_m/2], [-width_m/2, -height_m/2]])
        self.reference_to_world = cv2.getPerspectiveTransform(self.corners, target)
        self.board_area = abs(cv2.contourArea(self.corners))
        mask = np.zeros(reference.shape[:2], np.uint8)
        cv2.fillConvexPoly(mask, self.corners.astype(np.int32), 255)
        for p in self.corners:
            cv2.circle(mask, tuple(p.astype(int)), 40, 255, -1)
        if exclusion_polygon is not None:
            cv2.fillPoly(mask, [np.asarray(exclusion_polygon, np.int32)], 0)
        self.detector = cv2.ORB_create(nfeatures=2500, fastThreshold=10)
        self.keypoints, self.descriptors = self.detector.detectAndCompute(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), mask)
        if self.descriptors is None or len(self.keypoints) < min_inliers:
            raise ValueError("Insufficient static reference texture for moving-camera correction.")
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.min_inliers, self.min_coverage = min_inliers, min_coverage
        self.reference_gray = cv2.cvtColor(reference,cv2.COLOR_BGR2GRAY)
        self.reference_mask = mask
        self.sift = None

    def estimate(self, frame):
        matrix,stats = self._estimate_features(frame,self.detector,self.keypoints,self.descriptors,self.matcher,"ORB")
        if matrix is not None:
            return matrix,stats
        # Faint pen marks may have little ORB support outside a single edge.
        # An independent descriptor can recover spatial support; thresholds stay fixed.
        if self.sift is None:
            self.sift=cv2.SIFT_create(nfeatures=1800,contrastThreshold=.008)
            self.sift_keypoints,self.sift_descriptors=self.sift.detectAndCompute(self.reference_gray,self.reference_mask)
        if self.sift_descriptors is None:
            return matrix,stats
        fallback,fallback_stats=self._estimate_features(frame,self.sift,self.sift_keypoints,
                                                       self.sift_descriptors,cv2.BFMatcher(cv2.NORM_L2),"SIFT")
        fallback_stats['orb_inliers']=stats['inliers']
        fallback_stats['orb_coverage']=stats['coverage']
        return fallback,fallback_stats

    def _estimate_features(self,frame,detector,reference_keypoints,reference_descriptors,matcher,name):
        keypoints, descriptors = detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        stats = {"valid": False, "matches": 0, "inliers": 0, "coverage": 0., "median_reprojection_px": None,
                 "feature_detector":name}
        if descriptors is None:
            return None, stats
        pairs = matcher.knnMatch(descriptors, reference_descriptors, k=2)
        # One vote per reference keypoint prevents duplicate matches inflating support.
        unique = {}
        for pair in pairs:
            if len(pair) != 2:
                continue
            a, b = pair
            if a.distance < .75*b.distance and (a.trainIdx not in unique or a.distance < unique[a.trainIdx].distance):
                unique[a.trainIdx] = a
        good = list(unique.values())
        stats["matches"] = len(good)
        if len(good) < self.min_inliers:
            return None, stats
        source = np.float32([keypoints[m.queryIdx].pt for m in good])
        dest = np.float32([reference_keypoints[m.trainIdx].pt for m in good])
        matrix, inliers = cv2.findHomography(source, dest, cv2.RANSAC, 3., maxIters=3000, confidence=.995)
        if matrix is None or not np.isfinite(matrix).all() or inliers is None:
            return None, stats
        valid = inliers.ravel().astype(bool)
        stats["inliers"] = int(valid.sum())
        if valid.sum() < self.min_inliers:
            return None, stats
        hull = cv2.convexHull(dest[valid])
        stats["coverage"] = float(abs(cv2.contourArea(hull))/self.board_area)
        projected = cv2.perspectiveTransform(source[valid][None], matrix)[0]
        stats["median_reprojection_px"] = float(np.median(np.linalg.norm(projected-dest[valid], axis=1)))
        try:
            current_corners = cv2.perspectiveTransform(self.corners[None], np.linalg.inv(matrix))[0]
        except np.linalg.LinAlgError:
            return None, stats
        ratio = abs(cv2.contourArea(current_corners))/self.board_area
        if not cv2.isContourConvex(current_corners) or not .45 < ratio < 2.2:
            return None, stats
        if stats["coverage"] < self.min_coverage or stats["median_reprojection_px"] > 2.5:
            return None, stats
        stats["valid"] = True
        stats["corner_displacement_px"] = float(np.linalg.norm(current_corners-self.corners, axis=1).mean())
        stats["current_corners_px"] = current_corners.tolist()
        return self.reference_to_world @ matrix, stats

    def markers(self, frame, to_world, object_hsv, tool_hsv, size_ranges=None,tracking_margin_m=.03):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # A body can straddle a calibration line while remaining on the same table.
        # Do not crop its silhouette to the marked rectangle and select a hand instead.
        w, h = self.width_m*.5+tracking_margin_m, self.height_m*.5+tracking_margin_m
        interior = np.float32([[[-w,h], [w,h], [w,-h], [-w,-h]]])
        projected = cv2.perspectiveTransform(interior, np.linalg.inv(to_world))[0]
        roi = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(roi, projected.astype(np.int32), 255)
        result = []
        for role,ranges in enumerate((object_hsv, tool_hsv)):
            mask = np.zeros(frame.shape[:2], np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
            mask &= roi
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            count, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
            if count < 2:
                result.append(None)
                continue
            selected=None
            for best in np.argsort(stats[1:,cv2.CC_STAT_AREA])[::-1]+1:
                area = int(stats[best,cv2.CC_STAT_AREA])
                if area < 45:
                    break
                center_px = centers[best]
                center = cv2.perspectiveTransform(np.float32([[center_px]]), to_world)[0,0]
                contour, _ = cv2.findContours(np.uint8(labels == best)*255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                outline = max(contour, key=cv2.contourArea)
                outline_m = cv2.perspectiveTransform(outline.astype(np.float32), to_world)
                (_, _), (diam1,diam2), _ = cv2.minAreaRect(outline_m)
                if max(diam1,diam2)/max(min(diam1,diam2),1e-8)>1.8:
                    continue
                if size_ranges is not None:
                    low,high=size_ranges[role]
                    if min(diam1,diam2)<low or max(diam1,diam2)>high:
                        continue
                selected={"xy": center, "pixel_xy": center_px,"area_px": area,
                          "silhouette_diameter_m": float((diam1+diam2)/2),
                          "outside_calibration_rectangle":bool(abs(center[0])>self.width_m/2 or abs(center[1])>self.height_m/2)}
                break
            result.append(selected)
        return result


def track_planar_video(video, config, overlay_path=None,geometry_cache=None):
    """Return raw metric tracks, geometry diagnostics and optional streaming QA.

    Inputs need an explicit metric calibration. No origin/collector declaration is
    invented here. Missing marker or camera estimates remain NaN for later QA.
    """
    import imageio.v2 as imageio
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    hz = float(config.get("sample_hz", 10))
    if not cap.isOpened() or not 0 < hz <= fps:
        cap.release()
        raise ValueError("Invalid analysis video or sampling rate.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, config.get("reference_frame", 30))
    ok, reference = cap.read()
    if not ok:
        cap.release()
        raise ValueError("Cannot decode the selected reference frame.")
    tracker = PlanarTracker(reference, config["corners_px"], config["width_m"], config["height_m"],
                            config.get("reference_exclusion_polygon"),
                            config.get("min_inliers", 18), config.get("min_coverage", .12))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    times, objects, tools, diagnostics, silhouettes = [], [], [], [], []
    writer = imageio.get_writer(overlay_path, fps=hz, macro_block_size=2) if overlay_path else None
    index, next_time = 0, 0.
    world_to_view = np.array([[360/config["width_m"],0,180],
                              [0,-360/config["height_m"],180],[0,0,1]], dtype=float)
    world_corners=np.float32([[-config['width_m']/2,config['height_m']/2],
                             [config['width_m']/2,config['height_m']/2],
                             [config['width_m']/2,-config['height_m']/2],
                             [-config['width_m']/2,-config['height_m']/2]])
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current_time = index/fps
            index += 1
            if current_time+1e-7 < next_time:
                continue
            if geometry_cache is None:
                to_world, geometry = tracker.estimate(frame)
            else:
                if len(times)>=len(geometry_cache):
                    raise ValueError('Cached geometry does not match the analysis video sample count.')
                geometry=geometry_cache[len(times)]
                to_world=cv2.getPerspectiveTransform(np.float32(geometry['current_corners_px']),world_corners) if geometry['valid'] else None
            found = tracker.markers(frame, to_world, config["object_hsv"], config["tool_hsv"],
                                     config.get('marker_size_ranges_m'),config.get('tracking_margin_m',.03)) if to_world is not None else [None,None]
            times.append(current_time)
            for value, store in zip(found,(objects,tools)):
                store.append(value["xy"] if value else [np.nan,np.nan])
            silhouettes.append([value["silhouette_diameter_m"] if value else np.nan for value in found])
            diagnostics.append(geometry)
            if writer is not None:
                left = frame.copy()
                right = np.full((360,360,3),35,np.uint8)
                if to_world is not None:
                    right = cv2.warpPerspective(frame,world_to_view@to_world,(360,360))
                    corners=np.int32(geometry["current_corners_px"])
                    cv2.polylines(left,[corners],True,(0,220,220),2)
                    for marker,colour in zip(found,((0,0,255),(255,120,0))):
                        if marker:
                            cv2.circle(left,tuple(marker["pixel_xy"].astype(int)),9,colour,2)
                            point=world_to_view@np.r_[marker["xy"],1]
                            cv2.circle(right,tuple(point[:2].astype(int)),5,colour,2)
                else:
                    cv2.putText(right,"CAMERA FIT REJECTED",(12,180),0,.55,(0,0,255),1)
                for history,colour in ((objects,(0,0,255)),(tools,(255,130,0))):
                    array=np.asarray(history)
                    for a,b in zip(array[:-1],array[1:]):
                        if np.isfinite(a).all() and np.isfinite(b).all():
                            pa=(world_to_view@np.r_[a,1])[:2].astype(int)
                            pb=(world_to_view@np.r_[b,1])[:2].astype(int)
                            cv2.line(right,tuple(pa),tuple(pb),colour,1,cv2.LINE_AA)
                canvas=np.full((400,1000,3),(32,24,18),np.uint8)
                canvas[40:,:640]=cv2.resize(left,(640,360));canvas[40:,640:]=right
                text=f"RECORDED DATA / TRACKING QA | t={current_time:.1f}s | inliers={geometry['inliers']}"
                cv2.putText(canvas,text,(8,25),0,.55,(255,255,255),1,cv2.LINE_AA)
                writer.append_data(cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
            next_time += 1/hz
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    if geometry_cache is not None and len(times)!=len(geometry_cache):
        raise ValueError('Cached geometry has unmatched analysis samples.')
    return {"time":np.asarray(times),"object_xy":np.asarray(objects),"tool_xy":np.asarray(tools),
            "silhouette_diameters_m":np.asarray(silhouettes),"geometry":diagnostics,
            "reference_features":len(tracker.keypoints)}

