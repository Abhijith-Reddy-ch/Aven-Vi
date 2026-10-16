import time
from typing import List, Optional
from aven_vi.utils.types import TrackState
from aven_vi.audio.alert_priority import AlertPriority, AudioAlert
from aven_vi.audio.speech_generator import SpeechGenerator
from aven_vi.audio.earcon_generator import EarconGenerator
from aven_vi.audio.tts_dispatcher import AudioDispatcher

class AudioCoordinator:
    """
    High-level facade coordinating spatial perception tracks with acoustic guidance.
    Prioritizes obstacles, generates natural speech and spatial earcons,
    and manages anti-spam throttling without blocking video processing.
    """
    def __init__(
        self,
        enabled: bool = True,
        mode: str = "speech",  # "speech", "earcon", "hybrid"
        speech_rate: int = 175,
        volume: float = 0.90,
        alert_cooldown_s: float = 3.5,
        clear_pathway_interval_s: float = 6.0,
        mock_mode: bool = False
    ):
        self.enabled = bool(enabled)
        self.mode = str(mode).lower()
        self.clear_pathway_interval_s = float(clear_pathway_interval_s)
        self._last_corridor_hazard_time = time.time()
        self._last_clear_alert_time = 0.0

        self.dispatcher = AudioDispatcher(
            enabled=self.enabled,
            speech_rate=speech_rate,
            volume=volume,
            alert_cooldown_s=alert_cooldown_s,
            mock_mode=mock_mode
        )

    def select_primary_hazard(self, tracks: List[TrackState]) -> Optional[TrackState]:
        """
        Selects the single most safety-critical track from active tracks:
        1. In-corridor tracks ordered by distance (closest first).
        2. Immediate proximity obstacles (< 3.0 m) ordered by distance.
        3. Closest remaining tracked obstacle.
        """
        if not tracks:
            return None

        # Corridor obstacles have supreme priority
        corridor_tracks = [t for t in tracks if t.in_corridor]
        if corridor_tracks:
            return min(corridor_tracks, key=lambda t: t.estimated_distance_m)

        # Proximity obstacles within caution range (< 3.0 m)
        proximity_tracks = [t for t in tracks if t.estimated_distance_m < 3.0]
        if proximity_tracks:
            return min(proximity_tracks, key=lambda t: t.estimated_distance_m)

        # General nearest tracked object
        return min(tracks, key=lambda t: t.estimated_distance_m)

    def process_frame(self, tracks: List[TrackState], timestamp: Optional[float] = None) -> List[AudioAlert]:
        """
        Evaluates active tracks for the current video frame and dispatches
        appropriate speech and/or earcon guidance. Returns list of dispatched alerts.
        """
        if not self.enabled:
            return []

        ts = timestamp if timestamp is not None else time.time()
        dispatched_alerts = []

        primary_hazard = self.select_primary_hazard(tracks)

        if primary_hazard is not None:
            # If hazard is inside corridor or close, update corridor hazard timestamp
            if primary_hazard.in_corridor or primary_hazard.estimated_distance_m < 3.5:
                self._last_corridor_hazard_time = ts

            # 1. Earcon Audio Cue (in earcon or hybrid mode)
            if self.mode in ("earcon", "hybrid"):
                earcon = EarconGenerator.generate_earcon(primary_hazard, ts)
                if self.dispatcher.dispatch(earcon, distance_m=primary_hazard.estimated_distance_m):
                    dispatched_alerts.append(earcon)

            # 2. Spoken Speech Cue (in speech or hybrid mode)
            if self.mode in ("speech", "hybrid"):
                speech = SpeechGenerator.generate_alert(primary_hazard, ts)
                if self.dispatcher.dispatch(speech, distance_m=primary_hazard.estimated_distance_m):
                    dispatched_alerts.append(speech)

        else:
            # Pathway clear status check
            elapsed_clear = ts - self._last_corridor_hazard_time
            time_since_clear_msg = ts - self._last_clear_alert_time
            if (elapsed_clear >= self.clear_pathway_interval_s and 
                time_since_clear_msg >= self.clear_pathway_interval_s):
                clear_alert = SpeechGenerator.generate_clear_path_alert(ts)
                if self.dispatcher.dispatch(clear_alert):
                    dispatched_alerts.append(clear_alert)
                    self._last_clear_alert_time = ts

        return dispatched_alerts

    def stop(self):
        """Stops audio engine and shuts down background worker thread."""
        self.dispatcher.stop()
