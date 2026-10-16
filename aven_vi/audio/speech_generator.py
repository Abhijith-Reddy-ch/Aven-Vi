from aven_vi.utils.types import TrackState
from aven_vi.audio.alert_priority import AlertPriority, AudioAlert

class SpeechGenerator:
    """
    Translates 3D spatial track states into concise, natural spoken alerts
    tailored for blind and visually impaired users.
    """
    @staticmethod
    def generate_alert(track: TrackState, timestamp: float) -> AudioAlert:
        dist_m = track.estimated_distance_m
        dist_str = f"{dist_m:.1f} meters" if dist_m >= 1.0 else "one meter"
        clock = track.clock_position
        name = track.class_name.lower()
        elevation = track.elevation_zone

        # 1. Critical Emergency: in travel corridor and close (< 2.0 m)
        if track.in_corridor and dist_m < 2.0:
            priority = AlertPriority.CRITICAL_EMERGENCY
            if elevation == "GROUND":
                text = f"Stop! Low {name} in your path, {dist_str}."
            elif elevation == "HEAD":
                text = f"Stop! Overhead {name} directly ahead."
            else:
                text = f"Stop! {name.capitalize()} directly in your path, {dist_str}."

        # 2. Proximity Warning: in travel corridor (< 4.5 m) OR close non-corridor (< 2.5 m)
        elif track.in_corridor or dist_m < 2.5:
            priority = AlertPriority.PROXIMITY_WARNING
            if elevation == "GROUND":
                text = f"Caution, ground {name} at {clock}, {dist_str}."
            elif elevation == "HEAD":
                text = f"Caution, overhead {name} at {clock}, {dist_str}."
            elif track.in_corridor:
                text = f"Caution, {name} ahead in your path, {dist_str}."
            else:
                text = f"{name.capitalize()} at {clock}, {dist_str}."

        # 3. Informational Awareness: peripheral or distant
        else:
            priority = AlertPriority.AWARENESS_INFO
            text = f"{name.capitalize()} at {clock}, {dist_str}."

        return AudioAlert(
            priority=priority,
            text=text,
            track_id=track.track_id,
            created_at=timestamp,
            audio_type="SPEECH"
        )

    @staticmethod
    def generate_clear_path_alert(timestamp: float) -> AudioAlert:
        return AudioAlert(
            priority=AlertPriority.PATHWAY_STATUS,
            text="Pathway ahead is clear.",
            track_id=None,
            created_at=timestamp,
            audio_type="SPEECH"
        )
