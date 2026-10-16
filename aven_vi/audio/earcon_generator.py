from aven_vi.utils.types import TrackState
from aven_vi.audio.alert_priority import AlertPriority, AudioAlert

# Directional clock position to stereo panning mapping:
# -1.0 (Full Left) to +1.0 (Full Right)
CLOCK_PANNING_MAP = {
    "9:00": -1.0,
    "10:00": -0.65,
    "11:00": -0.30,
    "12:00": 0.0,
    "1:00": +0.30,
    "2:00": +0.65,
    "3:00": +1.0
}

class EarconGenerator:
    """
    Synthesizes spatial acoustic cues (earcons) with directional stereo panning
    and proximity-mapped acoustic frequencies.
    """
    @staticmethod
    def calculate_panning(clock_position: str) -> float:
        """Returns stereo pan value in [-1.0, +1.0]."""
        return CLOCK_PANNING_MAP.get(clock_position, 0.0)

    @staticmethod
    def calculate_frequency(distance_m: float) -> float:
        """
        Maps proximity to tone frequency (Hz):
        - Immediate danger (< 1.5m): 880 Hz (A5 - High warning pitch)
        - Caution zone (1.5 - 3.5m): 587 Hz (D5 - Medium pitch)
        - Safe distance (> 3.5m): 440 Hz (A4 - Standard awareness pitch)
        """
        if distance_m < 1.5:
            return 880.0
        elif distance_m < 3.5:
            return 587.3
        else:
            return 440.0

    @classmethod
    def generate_earcon(cls, track: TrackState, timestamp: float) -> AudioAlert:
        panning = cls.calculate_panning(track.clock_position)
        frequency = cls.calculate_frequency(track.estimated_distance_m)

        priority = AlertPriority.CRITICAL_EMERGENCY if track.in_corridor and track.estimated_distance_m < 2.0 else AlertPriority.PROXIMITY_WARNING

        return AudioAlert(
            priority=priority,
            text=f"BEEP at {track.clock_position} ({frequency:.0f}Hz)",
            track_id=track.track_id,
            created_at=timestamp,
            panning=panning,
            frequency=frequency,
            audio_type="EARCON"
        )
