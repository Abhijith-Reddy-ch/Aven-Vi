import pytest
from aven_vi.motion.trajectory import TrajectoryHistory
from aven_vi.utils.types import TrackState, ImageVelocity

def test_trajectory_history_limit():
    """Verify that TrajectoryHistory truncates at max_history and evicts oldest items."""
    max_len = 5
    traj = TrajectoryHistory(max_history=max_len)

    # Insert 10 points
    for i in range(10):
        traj.append(timestamp=float(i), cx=float(i * 10), cy=float(i * 20))

    # Length must be bounded at max_history (5)
    assert len(traj) == max_len

    points = traj.get_points()
    assert len(points) == 5

    # Oldest retained timestamp should be 5.0 (0-4 were evicted)
    assert points[0][0] == pytest.approx(5.0)
    assert points[-1][0] == pytest.approx(9.0)

    # Coordinates helper
    coords = traj.get_coordinates()
    assert coords[-1] == (pytest.approx(90.0), pytest.approx(180.0))

def test_trajectory_clear():
    traj = TrajectoryHistory(max_history=10)
    traj.append(1.0, 10.0, 20.0)
    traj.append(2.0, 15.0, 25.0)
    assert len(traj) == 2
    traj.clear()
    assert len(traj) == 0

def test_track_expiration_logic():
    """
    Simulate track lifecycle:
    Active track must be purged from memory when missing_frames > max_missing_frames.
    """
    max_missing = 3
    active_tracks = {
        1: TrackState(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(10, 10, 50, 50),
            centroid=(30.0, 30.0),
            first_seen=0.0,
            last_seen=0.0,
            age=1,
            missing_frames=0
        )
    }

    # Simulate 4 consecutive frames where track 1 is absent
    for frame_idx in range(1, 5):
        expired = []
        for tid, track in active_tracks.items():
            track.missing_frames += 1
            if track.missing_frames > max_missing:
                expired.append(tid)

        for tid in expired:
            del active_tracks[tid]

    # After 4 missing frames (exceeding max_missing=3), track 1 must be evicted
    assert 1 not in active_tracks
    assert len(active_tracks) == 0
