import numpy as np
import pytest

from scripts.analyze_external_control_topology import triangle_geometry


@pytest.mark.parametrize("scale,nonpositive", [(2.0, False), (-2.0, True), (0.0, True)])
def test_orientation_and_outside_queries(scale, nonpositive):
    source = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    target = source * [scale, 1]
    result = triangle_geometry(source, target, [[1.0, 1.0], [20.0, 20.0]])
    np.testing.assert_allclose(result["determinant"], scale)
    np.testing.assert_array_equal(result["query_nonpositive"], [nonpositive, False])
    np.testing.assert_array_equal(result["query_inside"], [True, False])


def test_control_queries_do_not_change_triangulation():
    controls = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    a = triangle_geometry(controls, controls, [[1.0, 1.0]])
    b = triangle_geometry(controls, controls, [[3.0, 4.0], [100.0, 100.0]])
    np.testing.assert_array_equal(a["triangles"], b["triangles"])
    np.testing.assert_array_equal(a["determinant"], b["determinant"])
