import numpy as np
import pytest

from phone2sim.data import Episode, load_episodes, split_episodes, transitions
from phone2sim.sim import PushEnv
from phone2sim.vision import centroid, fill_gaps, homography
from phone2sim.world_model import WorldModel


def episode(identifier="a", group=None, origin="synthetic"):
    t = np.arange(30)*.1
    obj = np.column_stack([np.linspace(0, .1, 30), np.zeros(30)])
    tool = obj - [.038, 0]
    return Episode(t, obj, tool, {"episode_id": identifier, "source_group": group or identifier, "origin": origin})


def test_episode_round_trip_does_not_require_pickle(tmp_path):
    e = episode()
    e.save(tmp_path/"e.npz")
    loaded = Episode.load(tmp_path/"e.npz")
    np.testing.assert_array_equal(loaded.object_xy, e.object_xy)
    assert loaded.metadata == e.metadata


def test_human_data_require_recording_provenance():
    with pytest.raises(ValueError, match="provenance"):
        episode(origin="human_video").validate()


def test_split_keeps_all_clips_of_same_source_together():
    episodes = [episode(str(i), str(i//2)) for i in range(12)]
    train, test = split_episodes(episodes)
    assert not ({e.metadata["source_group"] for e in train} & {e.metadata["source_group"] for e in test})
    assert len(train)+len(test) == 12


def test_mixed_origins_rejected(tmp_path):
    e = episode()
    e.save(tmp_path/"one.npz")
    # More useful failure than silently adding generated videos to the real split.
    human = episode("human", origin="human_video")
    human.metadata.update(collector="tester", collected_at="2026-10-20", source_sha256="x",
                          calibration_sha256="y", attested_own_recording=True)
    human.save(tmp_path/"two.npz")
    with pytest.raises(ValueError, match="mix"):
        load_episodes(tmp_path)


def test_calibration_metric_scale_and_y_up():
    import cv2
    h = homography([[0, 0], [400, 0], [400, 300], [0, 300]], .4, .3)
    result = cv2.perspectiveTransform(np.float32([[[200, 150], [300, 50]]]), h)[0]
    np.testing.assert_allclose(result, [[0, 0], [.1, .1]], atol=1e-6)
    with pytest.raises(ValueError, match="quadrilateral"):
        homography([[0, 0], [400, 300], [400, 0], [0, 300]], .4, .3)


def test_short_gap_interpolated_long_gap_rejected():
    track = np.column_stack([np.arange(30), np.zeros(30)]).astype(float)
    track[5] = np.nan
    result, missing = fill_gaps(track)
    assert missing == 1 and result[5, 0] == 5
    track[5:8] = np.nan
    with pytest.raises(ValueError, match="Occlusion"):
        fill_gaps(track)


def test_marker_detection_does_not_swap_red_and_blue():
    import cv2
    frame = np.zeros((120, 150, 3), np.uint8)
    cv2.circle(frame, (30, 40), 10, (0, 0, 255), -1)
    cv2.circle(frame, (100, 90), 8, (255, 0, 0), -1)
    red = centroid(frame, [[[0, 100, 65], [12, 255, 255]]])
    blue = centroid(frame, [[[92, 100, 65], [130, 255, 255]]])
    np.testing.assert_allclose(red, [30, 40]); np.testing.assert_allclose(blue, [100, 90])


def test_transition_velocity_is_causal():
    e = episode()
    before = transitions([e])
    e.object_xy[-1] += [1, 1]
    after = transitions([e])
    for i in range(4):
        np.testing.assert_allclose(before[i], after[i])
    assert not np.allclose(before[4], after[4])


def test_world_model_is_rotation_and_translation_equivariant():
    model = WorldModel().fit([episode()])
    obj = np.array([[.03, .02]])
    tool = obj-[.038, 0]
    vel, action = np.array([[.003, 0]]), np.array([[.006, 0]])
    pred, _, _ = model.predict(obj, tool, vel, action)
    rotation = np.array([[0, -1], [1, 0]])
    shift = np.array([.14, -.07])
    turned, _, _ = model.predict(obj@rotation.T+shift, tool@rotation.T+shift,
                                  vel@rotation.T, action@rotation.T)
    np.testing.assert_allclose(turned, pred@rotation.T, atol=1e-10)
    distant, _, _ = model.predict(obj, tool+[.2, .2], np.zeros((1, 2)), action)
    np.testing.assert_allclose(distant, 0)


def test_joint_servo_and_contact_move_puck_without_teleport():
    env = PushEnv()
    try:
        initial = env.reset([0, 0], [-.06, 0])
        for _ in range(14):
            current = env.step([.008, 0])
        assert current[0][0] > .025
        # A round puck can slip sideways; it must remain supported by the table.
        assert .010 < env.data.body("puck").xpos[2] < .020
        assert np.linalg.norm(current[1]-initial[1]) > .07
        assert np.isfinite(env.data.qpos).all()
        env.reset([0, 0], [-.06, .10])
        for _ in range(14):
            no_contact = env.step([.008, 0])
        assert np.linalg.norm(no_contact[0]) < 1e-4
    finally:
        env.close()


def test_synthetic_data_cannot_pass_submission_audit(tmp_path):
    from phone2sim.experiment import audit, train
    data = tmp_path/"data"
    data.mkdir()
    for i in range(4):
        episode(str(i)).save(data/f"{i}.npz")
    with pytest.raises(ValueError, match="allow-synthetic"):
        train(data, tmp_path/"rejected")
    train(data, tmp_path/"run", allow_synthetic=True)
    result = audit(data, tmp_path/"run")
    assert result["data_and_evaluation_checks_pass"] is False
    assert any("synthetic" in issue for issue in result["issues"])


def test_encoded_video_extracts_metric_tracks_without_claiming_human_origin(tmp_path):
    import cv2
    from phone2sim.data import write_json
    from phone2sim.vision import extract_tracks
    video = tmp_path/"synthetic_markers.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    assert writer.isOpened()
    try:
        for i in range(90):
            frame = np.full((480, 640, 3), 100, np.uint8)
            cv2.circle(frame, (200+i, 240), 20, (0, 0, 255), -1)
            cv2.circle(frame, (150+i, 240), 12, (255, 0, 0), -1)
            writer.write(frame)
    finally:
        writer.release()
    calibration = tmp_path/"calibration.json"
    write_json(calibration, {"corners_px": [[0,0], [640,0], [640,480], [0,480]],
                            "width_m": .4, "height_m": .3, "frame_size": [640,480],
                            "object_hsv": [[[0,100,65], [12,255,255]]],
                            "tool_hsv": [[[92,100,65], [130,255,255]]]})
    t, obj, tool, stats, previews = extract_tracks(video, calibration)
    assert len(t) == 30 and stats["interpolated_object_samples"] == 0
    np.testing.assert_allclose(obj[:, 1], 0, atol=.001)
    np.testing.assert_allclose(obj[0, 0], -.075, atol=.001)
    np.testing.assert_allclose(obj[-1, 0]-obj[0, 0], 87*.4/640, atol=.001)
    assert np.mean(np.linalg.norm(obj-tool, axis=1)) > .030
    assert len(previews) == len(t)


def test_dataset_change_refuses_evaluation(tmp_path):
    from phone2sim.experiment import train, verify_inputs
    data = tmp_path/"data"
    data.mkdir()
    for i in range(4):
        episode(str(i)).save(data/f"{i}.npz")
    _, _, manifest = train(data, tmp_path/"run", allow_synthetic=True)
    altered = episode("0")
    altered.object_xy[5, 0] += .01
    altered.save(data/"0.npz")
    with pytest.raises(ValueError, match="changed after training"):
        verify_inputs(data, manifest)


def test_interpolated_observations_do_not_enter_transition_learning_or_scoring(tmp_path):
    e=episode()
    e.observation_valid=np.ones(len(e.time),dtype=bool)
    e.observation_valid[15]=False
    e.object_xy[15]=[10,10]  # Deliberately absurd fill; every touching transition must be excluded.
    e.save(tmp_path/'episode.npz')
    loaded=Episode.load(tmp_path/'episode.npz')
    assert not loaded.valid_observations[15]
    obj,tool,velocity,action,target=transitions([loaded])
    assert len(target)==len(e.time)-2-3
    assert np.linalg.norm(target,axis=1).max()<.01
    assert np.linalg.norm(velocity,axis=1).max()<.01


def test_measured_geometry_is_preserved_in_checkpoint_and_simulator(tmp_path):
    e=episode()
    e.metadata.update(object_diameter_m=.045,tool_diameter_m=.0235)
    model=WorldModel().fit([e])
    model.save(tmp_path/'model.npz')
    loaded=WorldModel.load(tmp_path/'model.npz')
    assert loaded.object_radius==.0225
    assert loaded.tool_radius==.01175
    env=PushEnv(object_radius=loaded.object_radius,tool_radius=loaded.tool_radius)
    try:
        assert env.model.geom('puck_geom').size[0]==.0225
        assert env.model.geom('tip').size[0]==.01175
        assert np.isfinite(env.model.body_inertia).all()
    finally:
        env.close()

