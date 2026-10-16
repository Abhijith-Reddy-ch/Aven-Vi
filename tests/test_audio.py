import pytest
import queue
import time
from aven_vi.utils.types import TrackState
from aven_vi.audio.alert_priority import AlertPriority, AudioAlert
from aven_vi.audio.speech_generator import SpeechGenerator
from aven_vi.audio.earcon_generator import EarconGenerator, CLOCK_PANNING_MAP
from aven_vi.audio.tts_dispatcher import AudioDispatcher
from aven_vi.audio.audio_coordinator import AudioCoordinator

def create_mock_track(
    track_id: int = 1,
    class_name: str = "person",
    clock_position: str = "12:00",
    elevation_zone: str = "BODY",
    estimated_distance_m: float = 2.5,
    in_corridor: bool = True
) -> TrackState:
    return TrackState(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        confidence=0.90,
        bbox=(100.0, 100.0, 200.0, 300.0),
        centroid=(150.0, 200.0),
        first_seen=0.0,
        last_seen=0.0,
        age=1,
        trajectory_history=[(0.0, 150.0, 200.0)],
        clock_position=clock_position,
        bearing_degrees=0.0,
        elevation_zone=elevation_zone,
        estimated_distance_m=estimated_distance_m,
        in_corridor=in_corridor
    )

def test_alert_priority_ordering():
    pq = queue.PriorityQueue()
    a_info = AudioAlert(priority=AlertPriority.AWARENESS_INFO, text="Info alert")
    a_crit = AudioAlert(priority=AlertPriority.CRITICAL_EMERGENCY, text="Emergency alert")
    a_warn = AudioAlert(priority=AlertPriority.PROXIMITY_WARNING, text="Warning alert")
    a_path = AudioAlert(priority=AlertPriority.PATHWAY_STATUS, text="Path clear")

    # Add in scrambled order
    pq.put(a_info)
    pq.put(a_path)
    pq.put(a_crit)
    pq.put(a_warn)

    # Highest priority (lowest numeric value) must pop first
    first = pq.get()
    assert first.priority == AlertPriority.CRITICAL_EMERGENCY
    assert first.text == "Emergency alert"

    second = pq.get()
    assert second.priority == AlertPriority.PROXIMITY_WARNING

    third = pq.get()
    assert third.priority == AlertPriority.AWARENESS_INFO

    fourth = pq.get()
    assert fourth.priority == AlertPriority.PATHWAY_STATUS

def test_speech_generator_critical_corridor():
    track = create_mock_track(
        class_name="car",
        estimated_distance_m=1.8,
        in_corridor=True,
        elevation_zone="BODY"
    )
    alert = SpeechGenerator.generate_alert(track, timestamp=10.0)
    assert alert.priority == AlertPriority.CRITICAL_EMERGENCY
    assert "Stop! Car directly in your path, 1.8 meters." in alert.text
    assert alert.track_id == 1

def test_speech_generator_elevation_awareness():
    # Ground hazard
    ground_track = create_mock_track(
        class_name="dog",
        estimated_distance_m=1.2,
        in_corridor=True,
        elevation_zone="GROUND"
    )
    alert_g = SpeechGenerator.generate_alert(ground_track, timestamp=10.0)
    assert alert_g.priority == AlertPriority.CRITICAL_EMERGENCY
    assert "Low dog in your path" in alert_g.text

    # Overhead hazard
    head_track = create_mock_track(
        class_name="branch",
        estimated_distance_m=1.5,
        in_corridor=True,
        elevation_zone="HEAD"
    )
    alert_h = SpeechGenerator.generate_alert(head_track, timestamp=10.0)
    assert alert_h.priority == AlertPriority.CRITICAL_EMERGENCY
    assert "Overhead branch directly ahead" in alert_h.text

def test_earcon_panning_and_frequencies():
    assert EarconGenerator.calculate_panning("9:00") == -1.0
    assert EarconGenerator.calculate_panning("12:00") == 0.0
    assert EarconGenerator.calculate_panning("3:00") == 1.0
    assert EarconGenerator.calculate_panning("10:00") == -0.65
    assert EarconGenerator.calculate_panning("2:00") == 0.65

    # Frequency mapping
    assert EarconGenerator.calculate_frequency(1.0) == 880.0
    assert EarconGenerator.calculate_frequency(2.5) == 587.3
    assert EarconGenerator.calculate_frequency(5.0) == 440.0

    track = create_mock_track(clock_position="1:00", estimated_distance_m=1.2)
    earcon = EarconGenerator.generate_earcon(track, timestamp=10.0)
    assert earcon.audio_type == "EARCON"
    assert earcon.panning == 0.30
    assert earcon.frequency == 880.0

def test_audio_dispatcher_anti_spam():
    disp = AudioDispatcher(
        enabled=True,
        alert_cooldown_s=3.0,
        mock_mode=True
    )
    alert1 = AudioAlert(
        priority=AlertPriority.PROXIMITY_WARNING,
        text="Person at 12:00",
        track_id=10
    )

    # Initial alert is accepted
    assert disp.dispatch(alert1, distance_m=3.0) is True

    # Immediate duplicate is suppressed
    assert disp.dispatch(alert1, distance_m=3.0) is False

    # Significant distance drop (> 35%) triggers alert even within cooldown
    # 3.0 * 0.65 = 1.95m. Distance 1.8m is a >35% drop.
    assert disp.dispatch(alert1, distance_m=1.8) is True

    # Emergency priority always preempts
    alert_crit = AudioAlert(
        priority=AlertPriority.CRITICAL_EMERGENCY,
        text="Stop! Person close",
        track_id=10
    )
    assert disp.dispatch(alert_crit, distance_m=1.5) is True

def test_audio_coordinator_hazard_selection():
    coord = AudioCoordinator(
        enabled=True,
        mode="speech",
        mock_mode=True
    )

    t_offpath = create_mock_track(track_id=1, in_corridor=False, estimated_distance_m=1.5)
    t_corridor_far = create_mock_track(track_id=2, in_corridor=True, estimated_distance_m=4.0)
    t_corridor_near = create_mock_track(track_id=3, in_corridor=True, estimated_distance_m=2.2)

    primary = coord.select_primary_hazard([t_offpath, t_corridor_far, t_corridor_near])
    assert primary is not None
    # Must select t_corridor_near because it is in the travel corridor and closer than t_corridor_far
    assert primary.track_id == 3

def test_audio_coordinator_modes():
    t = create_mock_track(track_id=1, estimated_distance_m=2.0)

    # Speech mode
    c_speech = AudioCoordinator(mode="speech", mock_mode=True)
    alerts_s = c_speech.process_frame([t], timestamp=10.0)
    assert len(alerts_s) == 1
    assert alerts_s[0].audio_type == "SPEECH"

    # Earcon mode
    c_earcon = AudioCoordinator(mode="earcon", mock_mode=True)
    alerts_e = c_earcon.process_frame([t], timestamp=10.0)
    assert len(alerts_e) == 1
    assert alerts_e[0].audio_type == "EARCON"

    # Hybrid mode
    c_hybrid = AudioCoordinator(mode="hybrid", mock_mode=True)
    alerts_h = c_hybrid.process_frame([t], timestamp=10.0)
    assert len(alerts_h) == 2
    types = {a.audio_type for a in alerts_h}
    assert types == {"SPEECH", "EARCON"}

def test_audio_coordinator_clear_pathway():
    coord = AudioCoordinator(
        enabled=True,
        mode="speech",
        clear_pathway_interval_s=5.0,
        mock_mode=True
    )
    t0 = 100.0
    # Process frame with no tracks at t0 (corridor clear starts)
    coord._last_corridor_hazard_time = t0

    # At t0 + 2s, should not alert clear
    alerts_early = coord.process_frame([], timestamp=t0 + 2.0)
    assert len(alerts_early) == 0

    # At t0 + 5.5s, should trigger clear pathway alert
    alerts_clear = coord.process_frame([], timestamp=t0 + 5.5)
    assert len(alerts_clear) == 1
    assert alerts_clear[0].priority == AlertPriority.PATHWAY_STATUS
    assert "Pathway ahead is clear" in alerts_clear[0].text
