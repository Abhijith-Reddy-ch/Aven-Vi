from typing import List, Tuple
from collections import deque

class TrajectoryHistory:
    """
    Maintains a bounded temporal trajectory of centroids: [(timestamp, cx, cy), ...]
    Prevents unbounded memory growth by evicting older points when max_history is reached.
    """
    def __init__(self, max_history: int = 30):
        self.max_history = max(1, int(max_history))
        self._history: deque = deque(maxlen=self.max_history)

    def append(self, timestamp: float, cx: float, cy: float) -> None:
        """Appends a new centroid observation with its timestamp."""
        self._history.append((float(timestamp), float(cx), float(cy)))

    def get_points(self) -> List[Tuple[float, float, float]]:
        """Returns the chronological history as a list of (timestamp, cx, cy)."""
        return list(self._history)

    def get_coordinates(self) -> List[Tuple[float, float]]:
        """Returns list of (cx, cy) pixel points for drawing polylines."""
        return [(cx, cy) for _, cx, cy in self._history]

    def clear(self) -> None:
        """Clears all trajectory history."""
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)
