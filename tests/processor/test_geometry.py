import struct

import numpy as np
from atl06_processor.geometry import bbox_struct, points_to_wkb


def test_points_to_wkb_roundtrip():
    wkb = points_to_wkb(np.array([10.5]), np.array([-70.25]))
    assert wkb[0].as_py() == struct.pack("<BIdd", 1, 1, 10.5, -70.25)


def test_bbox_struct_is_degenerate_point_box():
    b = bbox_struct(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert b.to_pylist()[1] == {"xmin": 2.0, "ymin": 4.0, "xmax": 2.0, "ymax": 4.0}
