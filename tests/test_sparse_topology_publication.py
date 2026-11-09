from dataclasses import replace

from vectoreyes.candidate import FlatFill
from vectoreyes.fit import Shape
from vectoreyes.refinement.region_sample import RegionSample
from vectoreyes.path_geometry import CornerAnchor, CompoundPathGeometry, parse_svg_paths


def test_component_publication_preserves_sparse_canonical_command_identity():
    from vectoreyes.refinement.topology_binding import normalize_working_topology

    d = 'M0 0 L10 0 L10 10 L0 10 Z M20 0 L30 0 L30 10 L20 10 Z'
    paths = parse_svg_paths(d, region_id='r1')
    paths = tuple(replace(path, path=replace(path.path, commands=tuple(
        replace(command, id=f'{path.path.id}.c{index * 3}')
        for index, command in enumerate(path.path.commands)))) for path in paths)
    paths = tuple(replace(path, anchors=(CornerAnchor(f'{path.path.id}.corner',
        path.path.commands[1].id, path.path.commands[1].id),)) for path in paths)
    cached = CompoundPathGeometry(paths, 'nonzero', 'pending component publication')
    source = RegionSample(1, Shape('path', {'d': d, 'canonical_path_geometry': cached}),
        FlatFill('#123456'), drawing_id='r1')

    (result,) = normalize_working_topology((source,))

    assert len(result.children) == 2
    for child, expected in zip(result.children, paths, strict=True):
        actual = child.current.params['canonical_path_geometry']
        assert actual.path.commands == expected.path.commands
        assert actual.anchors == expected.anchors
    assert source.is_leaf
