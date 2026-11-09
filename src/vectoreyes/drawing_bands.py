"""Source-bound band selection and canonical publication shared by every caller.

Only the numeric engine knows how to detect/reconstruct/join a band. This layer
owns displayed addresses, original-source receipts, immutable publication and
whole-scene safety. Constant normal width is certified on interiors, not joins.
"""

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from collections.abc import Mapping

import numpy as np

from .bands import BandRequest
from .drawing import _walk_regions
from .drawing_actions import BandSpanInput, SetBandAction
from .path_geometry import CornerAnchor, CubicCommand, PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry
from .path_topology import component_paths, replace_component_paths
from .drawing_recipe import EffectiveDrawingRecipe, GeometryEvidence, PaintEvidence
from .path_fit_evidence import FitPathCorrespondence
from .replay_state import ReplayState, publish_drawing_state


@dataclass(frozen=True)
class BandSourceContext:
    evidence: GeometryEvidence
    paint: PaintEvidence
    recipe: EffectiveDrawingRecipe
    corner_replays: tuple = ()
    engine: object | None = None


def bands_pass(objects, masks):
    """Legacy numerical inputs cannot prove original command ownership."""
    import warnings
    del objects, masks
    warnings.warn('Band refinement skipped: exact original source context is unavailable; use the retained drawing engine.',
        RuntimeWarning, stacklevel=2)
    return []


def validate_band_span(drawing, span: BandSpanInput):
    """Require an exact fitted side, never a guessed or corner-owned command run."""
    region = drawing.region(span.region)
    paths = component_paths(region.geometry) if region.is_leaf and isinstance(
        region.geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)) else ()
    matching = [path for path in paths if all(identity in {cmd.id for cmd in path.path.commands}
        for identity in span.between)]
    if len(matching) != 1:
        raise ValueError('band boundaries must belong to one concrete path')
    path = matching[0]
    ids = tuple(cmd.id for cmd in path.path.commands)
    start, end = (ids.index(identity) for identity in span.between)
    if end == 0 and path.path.commands[-1].end == path.path.start:
        end = len(ids) - 1
    # A run crossing M needs a split of a generated cubic at the retained M.
    # Refuse it rather than invent an endpoint/source binding.
    if start >= end:
        raise ValueError('band side crosses the retained path origin or is empty')
    corners = tuple(anchor for anchor in path.anchors if isinstance(anchor, CornerAnchor))
    if not any(anchor.exit_command_id == span.between[0] for anchor in corners) or not any(
            anchor.entry_command_id == span.between[1] for anchor in corners):
        raise ValueError('band side must be bounded by retained corner exits and entries')
    owned = ids[start + 1:end + 1]
    corner_owned = {identity for anchor in corners for identity in anchor.owned_command_ids}
    if corner_owned.intersection(owned):
        raise ValueError('band side consumes corner-owned commands')
    if any(set(anchor.command_ids).intersection(owned[:-1]) for anchor in path.anchors):
        raise ValueError('band side consumes a retained feature boundary')
    return path, owned


def _source_span(context: BandSourceContext, state: ReplayState, span: BandSpanInput):
    from .drawing_replay import _original_source_span, _path_record_for_display
    path, owned = validate_band_span(state.drawing, span)
    record = _path_record_for_display(context.evidence, state, span.region, path.path.id,
        list(context.corner_replays), context.recipe)
    points = _original_source_span(context.evidence, record, path, *span.between)
    # Coincident original nodes carry distinct source addresses but no geometry.
    # Removing zero-length numeric edges leaves the exact material polyline.
    points = points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]]
    if len(points) == 2:
        points = np.vstack([points[0], (points[0] + points[1]) / 2, points[1]])
    return path, owned, record, points


def _trim(points, start_fraction, end_fraction):
    """Trim by measured source arc length; never infer source sample indices."""
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    start, end = arc[-1] * start_fraction, arc[-1] * (1 - end_fraction)
    interior = points[(arc > start + 1e-9) & (arc < end - 1e-9)]
    endpoints = np.column_stack([np.interp([start, end], arc, points[:, axis]) for axis in (0, 1)])
    selected = np.vstack([endpoints[0], interior, endpoints[1]])
    if len(selected) == 2:
        selected = np.vstack([selected[0], (selected[0] + selected[1]) / 2, selected[1]])
    return tuple(map(tuple, selected))


def _numeric_selection(engine, action, sources, retained):
    serialization_margin = np.sqrt(2.) * 0.5e-6
    budget = action.max_error - serialization_margin
    if budget <= 0:
        raise ValueError('band source budget is smaller than SVG serialization precision')
    fractions = np.linspace(0, action.max_transition_fraction, action.transition_steps + 1)
    candidates, rejections = [], {}
    for swap in (False, True):
        first, second = sources[::-1] if swap else sources
        for left, right in ((fraction, fraction) for fraction in fractions):
                request = BandRequest(_trim(first, left, right), tuple(map(tuple, second)),
                    max_error=budget, width_tolerance=action.width_tolerance,
                    tangent_tolerance_degrees=action.tangent_tolerance_degrees,
                    min_support=action.min_support)
                reconstruction = engine.reconstruct_band(request, width=action.width)
                if not reconstruction.accepted:
                    rejections[reconstruction.reason] = rejections.get(reconstruction.reason, 0) + 1
                    continue
                source_pairs = (first, second)
                endpoint_pairs = retained[::-1] if swap else retained
                aligned = []
                for points, ends, curves in zip(source_pairs, endpoint_pairs,
                        (reconstruction.offsets.first, reconstruction.offsets.second), strict=True):
                    direct = np.linalg.norm(points[0] - curves[0][0]) + np.linalg.norm(points[-1] - curves[-1][-1])
                    reverse = np.linalg.norm(points[-1] - curves[0][0]) + np.linalg.norm(points[0] - curves[-1][-1])
                    aligned.append(ends[::-1] if reverse < direct else ends)
                whole = replace(request, first=tuple(map(tuple, first)), second=tuple(map(tuple, second)))
                try:
                    joined = engine.join_band(whole, reconstruction.offsets, endpoints=tuple(aligned))
                except ValueError as error:
                    reason = f'invalid_join: {error}'
                    rejections[reason] = rejections.get(reason, 0) + 1
                    continue
                if joined.accepted:
                    curves = (joined.first, joined.second)
                    errors = joined.source_errors
                    if swap:
                        curves, errors = curves[::-1], errors[::-1]
                    candidates.append((max(errors), sum(errors), left + right, curves,
                        {'source_errors': errors, 'width': reconstruction.width,
                         'transition_fractions': (float(left), float(right)), 'primary_side': int(swap),
                         'interior_error_bound': reconstruction.offsets.error_bound,
                         'svg_rounding_bound': float(serialization_margin)}))
                else:
                    rejections[joined.reason] = rejections.get(joined.reason, 0) + 1
    if not candidates:
        raise ValueError(f'band source rejection: {rejections}')
    return tuple((candidate[3], candidate[4]) for candidate in sorted(candidates, key=lambda item: item[:3]))


def _replace_sides(path, replacements):
    """Replace only admitted side runs; renumber and remap every retained anchor."""
    starts, consumed = {}, set()
    for owned, curves in replacements:
        if consumed.intersection(owned):
            raise ValueError('band sides overlap in command ownership')
        consumed.update(owned)
        starts[owned[0]] = (owned, curves)
    commands, old_to_new, generated = [], {}, {}
    for command in path.path.commands:
        if command.id in starts:
            owned, curves = starts[command.id]
            output = []
            for curve in curves:
                identity = f'{path.path.id}.c{len(commands)}'
                commands.append(CubicCommand(identity, curve[-1], curve[1], curve[2]))
                output.append(identity)
            old_to_new[owned[-1]] = output[-1]
            generated[owned] = tuple(output)
        elif command.id not in consumed:
            identity = f'{path.path.id}.c{len(commands)}'
            commands.append(replace(command, id=identity))
            old_to_new[command.id] = identity
    anchors = []
    for anchor in path.anchors:
        if isinstance(anchor, CornerAnchor):
            anchors.append(replace(anchor, entry_command_id=old_to_new[anchor.entry_command_id],
                exit_command_id=old_to_new[anchor.exit_command_id],
                owned_command_ids=tuple(old_to_new[item] for item in anchor.owned_command_ids)))
        else:
            anchors.append(replace(anchor, command_ids=tuple(old_to_new[item] for item in anchor.command_ids)))
    return PathGeometry(replace(path.path, commands=tuple(commands)), tuple(anchors)), old_to_new, generated


def _remap_receipt(record, updated, mapping, generated):
    from .drawing_recipe import _digest
    correspondence = record.correspondence
    spans = []
    for span in correspondence.spans:
        owned = set(span.output_command_ids)
        replacement = next((output for consumed, output in generated.items()
            if owned == set(consumed)), None)
        if replacement is None:
            if not all(item in mapping for item in span.output_command_ids):
                continue
            replacement = tuple(mapping[item] for item in span.output_command_ids)
        spans.append(replace(span, output_command_ids=replacement))
    if not spans:
        raise ValueError('band publication would retire every exact source span')
    correspondence = FitPathCorrespondence(
        tuple(replace(item, output_command_id=mapping[item.output_command_id])
            for item in correspondence.boundaries if item.output_command_id in mapping),
        tuple(spans), tuple(replace(item, output_command_id=mapping[item.output_command_id])
            for item in correspondence.endpoints if item.output_command_id in mapping))
    return replace(record, output_revision=_digest(updated), correspondence=correspondence)


def decode_band_receipt(value):
    """Restore the typed receipt from primitive immutable metadata, not geometry."""
    from .drawing_recipe import PathReplayCorrespondence, SourcePathBinding
    from .path_fit_evidence import FitBoundaryCorrespondence, FitSpanCorrespondence, FitEndpointCorrespondence
    correspondence = value['correspondence']
    return PathReplayCorrespondence(binding=SourcePathBinding(**value['binding']),
        output_revision=value['output_revision'], output_region_id=value['output_region_id'],
        output_region_ancestry=tuple(value['output_region_ancestry']), output_path_id=value['output_path_id'],
        correspondence=FitPathCorrespondence(
            tuple(FitBoundaryCorrespondence(**item) for item in correspondence['boundaries']),
            tuple(FitSpanCorrespondence(**item) for item in correspondence['spans']),
            tuple(FitEndpointCorrespondence(**item) for item in correspondence['endpoints'])))


def remap_band_receipts(previous_metadata, updated, receipt):
    """Compose a later executor's proven endpoint lineage with band source receipts."""
    retained = previous_metadata.get('band_path_correspondence', {})
    if not retained:
        return {}
    previous = previous_metadata.get('path_command_lineage')
    current = receipt.get('path_command_lineage')
    if previous is None or current is None or previous['input_d'] != current['input_d']:
        return {}
    mapping = {middle: current['commands'][original]
        for original, middle in previous['commands'].items() if original in current['commands']}
    result = {}
    for path in component_paths(updated.geometry):
        value = retained.get(path.path.id)
        if value is not None:
            result[path.path.id] = asdict(_remap_receipt(decode_band_receipt(value), path, mapping, {}))
    return result


def _guard_scene(before, after, affected):
    """Do not delegate source or scene safety to the detector's admission score."""
    from .geometry_native import region_path
    from .boundary_topology import certify_simple_boundaries
    indexed_before = {item.id: item for root in before.regions for item in _walk_regions(root)}
    indexed_after = {item.id: item for root in after.regions for item in _walk_regions(root)}
    leaves = [item for item in indexed_before.values() if item.is_leaf]
    prior = {item.id: region_path(item, indexed_before) for item in leaves}
    final = {item.id: region_path(indexed_after[item.id], indexed_after) for item in leaves}
    for identity in affected:
        certify_simple_boundaries(component_paths(indexed_after[identity].geometry))
        if final[identity].is_empty:
            raise ValueError('band replacement creates invalid topology')
    # Clone/source resolution can change a footprint without changing its own
    # geometry record. Check every resolved leaf's topology and every pair.
    for identity in final:
        signature = lambda value: (len(value.geoms), sum(len(poly.interiors) for poly in value.geoms))
        if signature(prior[identity]) != signature(final[identity]):
            raise ValueError('band replacement changes component or hole topology')
    for left, right in combinations(leaves, 2):
        if final[left.id].intersection(final[right.id]).area > prior[left.id].intersection(prior[right.id]).area + 1e-7:
            raise ValueError('band replacement introduces region overlap')


def set_band(state: ReplayState, action: SetBandAction, context: BandSourceContext) -> ReplayState:
    """Execute one exact public declaration with original-material source guards."""
    from .python_drawing_engine import PythonDrawingEngine
    for span in (action.first, action.second):
        _source_ownership(state, span, context)
    evidence = [_source_span(context, state, span) for span in (action.first, action.second)]
    sources = tuple(item[3] for item in evidence)
    retained = tuple((path.endpoint(span.between[0]), path.endpoint(span.between[1]))
        for (path, _, _, _), span in zip(evidence, (action.first, action.second), strict=True))
    rejected = []
    for curves, report in _numeric_selection(context.engine or PythonDrawingEngine(), action, sources, retained):
        try:
            return _publish_band_candidate(state, action, evidence, retained, curves,
                {**report, 'rejected_candidates': tuple(rejected)})
        except ValueError as error:
            rejected.append({**report, 'reason': str(error)})
    raise ValueError(f'no scene-safe band candidate: {rejected}')


def _publish_band_candidate(state, action, evidence, retained, curves, report):
    """Assess one complete candidate from the unchanged input; publish only on success."""
    from .drawing_relationships import replace_region
    replacements = {}
    for span, (path, owned, record, _), side, endpoints in zip(
            (action.first, action.second), evidence, curves, retained, strict=True):
        if side[0][0] != endpoints[0]:
            side = tuple(tuple(reversed(curve)) for curve in reversed(side))
        if side[0][0] != endpoints[0] or side[-1][-1] != endpoints[-1]:
            raise ValueError('band join did not preserve retained endpoints')
        key = (span.region, path.path.id)
        entry = replacements.setdefault(key, [path, record, []])
        entry[2].append((owned, side))
    drawing, metadata = state.drawing, {}
    for region_id in dict.fromkeys(key[0] for key in replacements):
        region = drawing.region(region_id)
        paths, immediate = [], {}
        receipts = dict(state.shape_metadata.get(region_id, {}).get('band_path_correspondence', {}))
        for path in component_paths(region.geometry):
            selected = replacements.get((region_id, path.path.id))
            if selected is None:
                paths.append(path)
                immediate.update({cmd.id: cmd.id for cmd in path.path.commands})
                continue
            updated, mapping, generated = _replace_sides(path, selected[2])
            paths.append(updated)
            immediate.update(mapping)
            receipts[path.path.id] = asdict(_remap_receipt(selected[1], updated, mapping, generated))
        geometry = replace_component_paths(region.geometry, tuple(paths))
        info = dict(state.shape_metadata.get(region_id, {}))
        prior = info.get('path_command_lineage')
        input_d = region.geometry.to_svg_d()
        lineage = immediate
        if isinstance(prior, Mapping):
            input_d = prior['input_d']
            lineage = {old: immediate[current] for old, current in prior['commands'].items() if current in immediate}
        info['path_command_lineage'] = {'input_d': input_d, 'output_d': geometry.to_svg_d(), 'commands': lineage}
        info['band_path_correspondence'] = receipts
        metadata[region_id] = info
        updated = replace(region, geometry=geometry,
            diagnostics={**region.diagnostics, 'bands': {'accepted': True, **report,
                'settings': action.model_dump(exclude_none=True), 'constant_width_scope': 'interior_only'}})
        drawing = replace_region(drawing, updated)
    _guard_scene(state.drawing, drawing, set(metadata))
    return publish_drawing_state(state, drawing, shape_metadata=metadata)


def discover_band_actions(state):
    """Enumerate only corner-to-corner fitted sides; source admission stays shared."""
    spans = []
    for root in state.drawing.regions:
        for leaf in root.leaves():
            if not isinstance(leaf.geometry, (PathGeometry, PathGeometryWithCutouts, CompoundPathGeometry)):
                continue
            for path in component_paths(leaf.geometry):
                corners = [anchor for anchor in path.anchors if isinstance(anchor, CornerAnchor)]
                ids = [cmd.id for cmd in path.path.commands]
                corners.sort(key=lambda anchor: ids.index(anchor.entry_command_id))
                for first, second in zip(corners, (*corners[1:], *corners[:1])):
                    span = BandSpanInput(region=leaf.id, between=(first.exit_command_id, second.entry_command_id))
                    try:
                        validate_band_span(state.drawing, span)
                    except ValueError:
                        continue
                    ends = np.asarray([path.endpoint(item) for item in span.between])
                    direction = ends[1] - ends[0]
                    length = float(np.linalg.norm(direction))
                    if length >= 32:
                        spans.append((span, direction / length, length))
    choices = []
    for (first, a, alen), (second, b, blen) in combinations(spans, 2):
        if abs(float(a @ b)) < 0.8 or min(alen, blen) / max(alen, blen) < 0.5:
            continue
        first_path, _ = validate_band_span(state.drawing, first)
        second_path, _ = validate_band_span(state.drawing, second)
        first_midpoint = np.mean([first_path.endpoint(item) for item in first.between], axis=0)
        second_midpoint = np.mean([second_path.endpoint(item) for item in second.between], axis=0)
        delta = second_midpoint - first_midpoint
        gap = abs(float(a[0] * delta[1] - a[1] * delta[0]))
        choices.append((gap, SetBandAction(op='set_band', first=first, second=second)))
    yield from (action for _, action in sorted(choices, key=lambda item: item[0]))


def _source_ownership(state, span, context):
    from .drawing_replay import _source_command_from_correspondence, _directed_source_owned_commands
    path, _, record, _ = _source_span(context, state, span)
    source = next(item.fit.source for item in context.evidence.root_paths
        if item.fit.source.path.id == record.binding.source_path_id)
    start, end = (_source_command_from_correspondence(record.correspondence,
        path.path.boundary_command_id(identity)) for identity in span.between)
    owned = _directed_source_owned_commands(source, start, end)
    corner_owned = {identity for anchor in source.anchors if isinstance(anchor, CornerAnchor)
        for identity in anchor.owned_command_ids}
    if corner_owned.intersection(owned):
        raise ValueError('band side consumes original source corner-owned commands')
    return frozenset((source.path.id, identity) for identity in owned)


def refine_band_regions(state, *, context: BandSourceContext | None, preceding_steps=()):
    """Discover declarations, select once, and retain ordinary operation outcomes."""
    from .drawing_proposals import CanonicalDetectionResult
    from .drawing_replay import bind_recipe_operation
    from .selected_outcomes import retain_selected_outcome, selected_input_revision
    if context is None:
        return CanonicalDetectionResult(state, (), (), diagnostics={
            'accepted': False, 'reason': 'original_source_context_unavailable'})
    steps, rejected, consumed, attempted = [], [], set(), set()
    while True:
        changed = False
        upstream = replace(context.recipe, steps=(*context.recipe.steps, *preceding_steps, *steps))
        current_context = replace(context, recipe=upstream)
        for action in discover_band_actions(state):
            try:
                ownership = frozenset().union(*(_source_ownership(state, span, current_context)
                    for span in (action.first, action.second)))
                if consumed.intersection(ownership) or ownership in attempted:
                    continue
                attempted.add(ownership)
                selected = set_band(state, action, current_context)
                operation = bind_recipe_operation(operation_id=f'auto.bands.{len(preceding_steps)}.{len(steps)}',
                    action=action, settings=upstream.settings, evidence_revision=context.evidence.revision, state=state)
                operation = replace(operation, origin='automatic', diagnostics={
                    'input_revision': selected_input_revision(state),
                    'band_settings': action.model_dump(exclude_none=True)})
                operation, selected = retain_selected_outcome(operation, state, selected,
                    evidence=context.evidence, paint=context.paint, upstream=upstream)
            except ValueError as error:
                rejected.append({'action': action.model_dump(exclude_none=True), 'reason': str(error)})
                continue
            state = selected
            steps.append(operation)
            consumed.update(ownership)
            changed = True
            break  # Rebind every remaining side against the new canonical addresses.
        if not changed:
            break
    return CanonicalDetectionResult(state, tuple(steps), (), diagnostics={
        'accepted': bool(steps), 'accepted_count': len(steps), 'rejections': tuple(rejected),
        'max_error': 1.5, 'constant_width_scope': 'interior_only'})
