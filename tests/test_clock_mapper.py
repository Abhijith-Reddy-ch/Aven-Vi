import pytest
from aven_vi.spatial.clock_mapper import ClockMapper

def test_clock_positions():
    mapper = ClockMapper(hfov_deg=70.0)
    w = 1000

    # 9:00 (Far left)
    pos, bearing = mapper.calculate_bearing(cx=50, frame_width=w)
    assert pos == "9:00"
    assert bearing < -25.0

    # 10:00 (Left flank)
    pos, bearing = mapper.calculate_bearing(cx=200, frame_width=w)
    assert pos == "10:00"
    assert -25.0 <= bearing < -15.0

    # 11:00 (Left ahead)
    pos, bearing = mapper.calculate_bearing(cx=350, frame_width=w)
    assert pos == "11:00"
    assert -15.0 <= bearing < -5.0

    # 12:00 (Direct ahead)
    pos, bearing = mapper.calculate_bearing(cx=500, frame_width=w)
    assert pos == "12:00"
    assert -5.0 <= bearing <= 5.0

    # 1:00 (Right ahead)
    pos, bearing = mapper.calculate_bearing(cx=650, frame_width=w)
    assert pos == "1:00"
    assert 5.0 < bearing <= 15.0

    # 2:00 (Right flank)
    pos, bearing = mapper.calculate_bearing(cx=800, frame_width=w)
    assert pos == "2:00"
    assert 15.0 < bearing <= 25.0

    # 3:00 (Far right)
    pos, bearing = mapper.calculate_bearing(cx=950, frame_width=w)
    assert pos == "3:00"
    assert bearing > 25.0

def test_corridor_check():
    # Central 30% means [0.35 * 1000, 0.65 * 1000] = [350, 650]
    mapper = ClockMapper(corridor_ratio=0.30)
    w = 1000

    # 1. Object dead center
    assert mapper.is_in_corridor((450, 100, 550, 500), frame_width=w) is True

    # 2. Object clearly on the left outside corridor [100, 250]
    assert mapper.is_in_corridor((100, 100, 250, 500), frame_width=w) is False

    # 3. Object clearly on the right outside corridor [700, 900]
    assert mapper.is_in_corridor((700, 100, 900, 500), frame_width=w) is False

    # 4. Object straddling the corridor boundary [300, 400]
    assert mapper.is_in_corridor((300, 100, 400, 500), frame_width=w) is True
