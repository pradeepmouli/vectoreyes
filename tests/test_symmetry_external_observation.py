import numpy as np

from vectormark.candidate import FlatFill
from vectormark.fit import Shape
from vectormark.optimizer.vector_region import VectorRegion


def test_symmetry_union_consumes_resolved_external_instance_without_local_source():
    from vectormark.optimizer.symmetry_materialize import union_self_symmetry

    outside = VectorRegion(1, Shape('rect', {'x': 0, 'y': 0, 'w': 10, 'h': 10}), FlatFill('#123456'))
    transform = (1, 0, 0, 1, 20, 0)
    clone = VectorRegion(3, Shape('use', {'href_obj_id': 1, 'href_region_path': ('..', 1),
        'transform': transform}), FlatFill('#123456'), footprint=outside.footprint.affine_transform(transform))
    source = VectorRegion.branch(id=2, children=(clone,), drawing_id='r2', raster=np.zeros((40, 40), bool))

    result = union_self_symmetry(source, (clone.footprint,), diagnostics={'mode': 'self'})

    assert result.is_leaf and result.footprint.area == 100
    assert result.fill is clone.fill
    assert result.footprint.bounds == (20, 0, 30, 10)
    assert source.is_branch and clone.current.kind == 'use'
