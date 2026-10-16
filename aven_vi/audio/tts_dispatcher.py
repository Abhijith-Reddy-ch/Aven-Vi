import threading
import queue
import time
from typing import Dict, Tuple, Optional
from aven_vi.audio.alert_priority import AudioAlert, AlertPriority

class AudioDispatcher:
    """
    Thread-safe, non-blocking audio dispatching engine for AVEN-VI.
    Executes Text-to-Speech and audio cues in a dedicated worker thread with alert prioritization,
    emergency preemption, and anti-spam repetition throttling.
    """
    def __init__(
        self,
        enabled: bool = True,
        speech_rate: int = 175,
        volume: float = 0.90,
        alert_cooldown_s: float = 3.5,
        mock_mode: bool = False
    ):
        self.enabled = bool(enabled)
        self.speech_rate = int(speech_rate)
        self.volume = float(volume)
        self.alert_cooldown_s = float(alert_cooldown_s)
        self.mock_mode = bool(mock_mode)

        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        # Keyed by (track_id, audio_type) so earcons and speech don't suppress each other in hybrid mode
        self._last_spoken_time: Dict[Tuple[int, str], float] = {}
        self._last_spoken_distance: Dict[Tuple[int, str], float] = {}
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self.spoken_log = []  # Record for auditing and unit tests

        if self.enabled:
            self._start_worker()

    def _start_worker(self):
        self._worker_thread = threading.Thread(target=self._speech_loop, daemon=True)
        self._worker_thread.start()

    def _speech_loop(self):
        # Initialize pyttsx3 inside the worker thread to satisfy Windows COM threading models
        engine = None
        if not self.mock_mode:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.speech_rate)
                engine.setProperty("volume", self.volume)
            except Exception as e:
                print(f"[WARN] TTS engine failed to initialize: {e}. Falling back to mock audio.")
                self.mock_mode = True

        while not self._stop_event.is_set():
            try:
                alert: AudioAlert = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            self.spoken_log.append(alert)

            if engine is not None and not self.mock_mode:
                try:
                    if alert.audio_type == "SPEECH":
                        engine.say(alert.text)
                        engine.runAndWait()
                    elif alert.audio_type == "EARCON":
                        try:
                            import winsound
                            winsound.Beep(int(alert.frequency), 120)
                        except Exception:
                            pass
                except Exception as ex:
                    print(f"[WARN] Audio playback error: {ex}")

            self._queue.task_done()

    def dispatch(self, alert: AudioAlert, distance_m: Optional[float] = None) -> bool:
        """
        Submits an alert to the priority queue after anti-spam throttling checks.
        Returns True if accepted into the queue, False if suppressed.
        """
        if not self.enabled:
            return False

        now = time.time()
        tid = alert.track_id
        tkey = (tid, alert.audio_type) if tid is not None else None

        # Emergency preemption: critical alerts bypass normal cooldown
        if alert.priority == AlertPriority.CRITICAL_EMERGENCY:
            if tkey is not None:
                self._last_spoken_time[tkey] = now
                if distance_m is not None:
                    self._last_spoken_distance[tkey] = distance_m
            self._queue.put(alert)
            return True

        # Anti-Spam Throttling for repeated tracks within same audio modality
        if tkey is not None and tkey in self._last_spoken_time:
            elapsed = now - self._last_spoken_time[tkey]
            last_dist = self._last_spoken_distance.get(tkey, 999.0)

            # Suppress if within cooldown, unless distance dropped significantly (> 35%)
            if elapsed < self.alert_cooldown_s:
                if distance_m is None or distance_m >= (last_dist * 0.65):
                    return False

        if tkey is not None:
            self._last_spoken_time[tkey] = now
            if distance_m is not None:
                self._last_spoken_distance[tkey] = distance_m

        self._queue.put(alert)
        return True

    def stop(self):
        """Stops the worker thread and flushes remaining queue items into spoken_log."""
        # Drain remaining queued alerts into spoken_log before terminating
        time.sleep(0.15)
        while not self._queue.empty():
            try:
                alert = self._queue.get_nowait()
                if alert not in self.spoken_log:
                    self.spoken_log.append(alert)
                self._queue.task_done()
            except queue.Empty:
                break

        self._stop_event.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
