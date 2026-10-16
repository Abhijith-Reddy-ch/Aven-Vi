from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

class AlertPriority(IntEnum):
    """
    Alert urgency hierarchy for visually impaired navigation.
    Lower numerical values represent higher urgency (Priority 1 preempts Priority 2/3).
    """
    CRITICAL_EMERGENCY = 1   # Immediate collision hazard in direct path (< 2.0 m)
    PROXIMITY_WARNING = 2    # Approaching obstacle within caution zone (2.0 - 4.5 m)
    AWARENESS_INFO = 3       # Peripheral object off-path or distant (> 4.5 m)
    PATHWAY_STATUS = 4       # Periodic confirmation (e.g., "Pathway is clear")

@dataclass(order=True)
class AudioAlert:
    """
    Priority-queued audio alert payload.
    The 'sort_index' ensures PriorityQueue pops highest urgency alerts first.
    """
    sort_index: int = field(init=False)
    priority: AlertPriority
    text: str
    track_id: Optional[int] = None
    created_at: float = 0.0
    panning: float = 0.0        # -1.0 (full left) to +1.0 (full right)
    frequency: float = 440.0    # Tone pitch in Hz (for earcons)
    audio_type: str = "SPEECH"  # "SPEECH" or "EARCON"

    def __post_init__(self):
        self.sort_index = int(self.priority)
