import cv2
import numpy as np

from phone2sim.planar_tracking import PlanarTracker


def textured_board():
    image=np.full((600,800,3),210,np.uint8)
    rng=np.random.default_rng(19)
    for _ in range(450):
        x,y=rng.integers([65,65],[735,535])
        if 270<x<530 and 200<y<400:
            continue
        shade=int(rng.integers(15,140))
        cv2.circle(image,(int(x),int(y)),int(rng.integers(1,4)),(shade,shade,shade),-1)
    corners=np.float32([[60,60],[740,60],[740,540],[60,540]])
    return image,corners


def test_camera_translation_does_not_become_object_motion():
    background,corners=textured_board()
    reference=background.copy()
    cv2.circle(reference,(420,300),20,(0,0,255),-1)
    cv2.circle(reference,(360,300),13,(255,0,0),-1)
    tracker=PlanarTracker(reference,corners,.4,.3,[[270,200],[530,200],[530,400],[270,400]])
    camera=np.array([[1.01,.012,19],[-.008,1.01,-12],[.000015,-.00001,1]],float)
    current=cv2.warpPerspective(reference,camera,(800,600),borderValue=(30,30,30))
    matrix,stats=tracker.estimate(current)
    assert stats['valid'] and stats['inliers']>30
    red,blue=tracker.markers(current,matrix,[[[0,150,45],[8,255,255]]],[[[92,110,35],[135,255,255]]])
    expected=cv2.perspectiveTransform(np.float32([[[420,300],[360,300]]]),tracker.reference_to_world)[0]
    np.testing.assert_allclose(red['xy'],expected[0],atol=.0015)
    np.testing.assert_allclose(blue['xy'],expected[1],atol=.0015)


def test_real_object_motion_survives_camera_correction():
    background,corners=textured_board()
    reference=background.copy()
    cv2.circle(reference,(420,300),20,(0,0,255),-1)
    tracker=PlanarTracker(reference,corners,.4,.3,[[270,200],[530,200],[530,400],[270,400]])
    moved=background.copy()
    cv2.circle(moved,(470,300),20,(0,0,255),-1)
    camera=np.array([[1,0,-21],[0,1,17],[0,0,1]],float)
    current=cv2.warpPerspective(moved,camera,(800,600))
    matrix,stats=tracker.estimate(current)
    assert stats['valid']
    red,_=tracker.markers(current,matrix,[[[0,150,45],[8,255,255]]],[[[92,110,35],[135,255,255]]])
    expected=cv2.perspectiveTransform(np.float32([[[470,300]]]),tracker.reference_to_world)[0,0]
    np.testing.assert_allclose(red['xy'],expected,atol=.0015)


def test_failed_camera_fit_is_rejected_instead_of_freezing_previous_geometry():
    reference,corners=textured_board()
    tracker=PlanarTracker(reference,corners,.4,.3)
    matrix,stats=tracker.estimate(np.full_like(reference,180))
    assert matrix is None and not stats['valid']


def test_independent_descriptor_fallback_recovers_valid_global_geometry():
    reference,corners=textured_board()
    tracker=PlanarTracker(reference,corners,.4,.3)
    class NoOrbFeatures:
        def detectAndCompute(self,*args):
            return [],None
    tracker.detector=NoOrbFeatures()
    camera=np.array([[1,0,12],[0,1,-9],[0,0,1]],float)
    current=cv2.warpPerspective(reference,camera,(800,600))
    matrix,stats=tracker.estimate(current)
    assert stats['valid'] and stats['feature_detector']=='SIFT'
    actual=cv2.perspectiveTransform(np.float32([[[412,291]]]),matrix)[0,0]
    expected=cv2.perspectiveTransform(np.float32([[[400,300]]]),tracker.reference_to_world)[0,0]
    np.testing.assert_allclose(actual,expected,atol=.0015)


def test_object_at_calibration_edge_is_not_cropped_in_favour_of_red_hand_blob():
    reference,corners=textured_board()
    tracker=PlanarTracker(reference,corners,.4,.3)
    frame=reference.copy()
    # The target straddles the top calibration line. A red elongated distractor
    # is deliberately larger in pixel area, as a coloured hand region can be.
    cv2.circle(frame,(420,65),32,(0,0,255),-1)
    cv2.rectangle(frame,(200,300),(225,480),(0,0,255),-1)
    red,_=tracker.markers(frame,tracker.reference_to_world,
                         [[[0,150,45],[8,255,255]]],[[[92,110,35],[135,255,255]]],
                         size_ranges=[[.03,.065],[.008,.035]])
    expected=cv2.perspectiveTransform(np.float32([[[420,65]]]),tracker.reference_to_world)[0,0]
    assert red is not None
    np.testing.assert_allclose(red['xy'],expected,atol=.0015)

