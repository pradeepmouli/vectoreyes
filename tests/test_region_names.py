from dataclasses import replace
import xml.etree.ElementTree as ET

import pytest
from pydantic import ValidationError

from vectoreyes.drawing import Canvas, Drawing, DrawingRegion
from vectoreyes.drawing_actions import DrawingPlan
from vectoreyes.drawing_fill import FlatFill
from vectoreyes.drawing_render import drawing_summary, render_drawing, render_labeled_drawing
from vectoreyes.path_geometry import parse_svg_paths
from tests.test_operation_outcomes import fixture
import vectoreyes.drawing_replay as replay


def region(rid='r1', name=None, d='M0 0 L8 0 L8 8 L0 8 Z', **kwargs):
    return DrawingRegion(rid, geometry=parse_svg_paths(d, region_id=rid)[0],
        fill=FlatFill('#123456'), name=name, **kwargs)


def submit(evidence, paint, recipe, ops):
    plan = DrawingPlan.model_validate(dict(version='vectoreyes.plan.v1', drawing_id='drw_test',
        base_version='v0', ops=ops))
    extension = replay.extend_recipe_with_plan(evidence=evidence, paint=paint,
        base_recipe=recipe, plan=plan)
    execution = replay.propose_effective_recipe(drawing_id='drw_test', evidence=evidence,
        paint=paint, recipe=extension.recipe)
    return extension.recipe, execution


def test_names_are_optional_lowercase_hyphenated_values():
    assert region(name='orange-band').name == 'orange-band'
    for invalid in ('Orange band', 'orange_band', '-orange', 'orange-', 'orange--band', ''):
        with pytest.raises(ValueError):
            region(name=invalid)


def test_name_then_address_in_same_plan_is_metadata_only_and_replays_by_id(monkeypatch):
    evidence, paint, base = fixture()
    before = replay.propose_effective_recipe(drawing_id='drw_test', evidence=evidence, paint=paint, recipe=base)
    recipe, output = submit(evidence, paint, base, [
        dict(op='set_region_name', target='r1', name='orange-band'),
        dict(op='set_fill', target={'name':'orange-band'}, fill=dict(type='flat', color='#abcdef'))])
    after = output.drawing.region('r1')
    assert after.name == 'orange-band'
    assert after.geometry == before.drawing.region('r1').geometry
    assert after.fill == FlatFill('#abcdef')
    assert recipe.operations[1].action.target == 'r1'
    assert not any(getattr(step, 'scope', None) == 'changed' for step in recipe.steps)
    renamed_recipe, renamed = submit(evidence, paint, recipe, [
        dict(op='set_region_name', target={'name':'orange-band'}, name='body-band')])
    assert renamed.drawing.region('r1').name == 'body-band'
    assert renamed.drawing.region('r1').fill == FlatFill('#abcdef')
    assert replay.reproduce_effective_recipe(drawing=renamed.drawing, evidence=evidence,
        paint=paint, recipe=renamed_recipe, replay=renamed.replay) == renamed.drawing


def test_name_address_reports_missing_and_ambiguous_candidates():
    from vectoreyes.region_addressing import resolve_region_address
    drawing = Drawing('d', Canvas(10,10), (region(name='band'), region('r2', name='band')))
    with pytest.raises(ValueError, match='ambiguous.*r1.*r2'):
        resolve_region_address({'name':'band'}, drawing)
    with pytest.raises(ValueError, match='not found'):
        resolve_region_address({'name':'missing'}, drawing)
    assert resolve_region_address('r1', drawing) == 'r1'


def test_names_label_groups_objects_and_maps_without_changing_ids():
    child = region(name='orange-band')
    drawing = Drawing('d', Canvas(10,10), (DrawingRegion('body', children=(child,), name='radish-body'),))
    report = drawing_summary(drawing)
    assert report['groups'][0]['name'] == 'radish-body'
    assert report['targets'][0]['name'] == 'orange-band'
    xml = ET.fromstring(render_drawing(drawing).svg)
    ns = '{http://www.inkscape.org/namespaces/inkscape}'
    parent = xml.find(".//*[@id='body']")
    assert parent.tag.endswith('g') and parent.get(ns+'label') == 'radish-body'
    child_element = parent.find(".//*[@id='r1']")
    assert child_element.get(ns+'label') == 'orange-band'
    assert child_element.find('{http://www.w3.org/2000/svg}title').text == 'orange-band'
    assert 'orange-band' in render_labeled_drawing(drawing)


def test_plain_svg_omits_editor_extensions_but_keeps_stable_ids():
    drawing = Drawing('d', Canvas(10,10), (region(name='orange-band'),))
    svg = render_drawing(drawing, editable=False).svg
    assert 'inkscape' not in svg and 'vectoreyes' not in svg
    assert ET.fromstring(svg).find(".//*[@id='r1']") is not None


def test_cutout_is_relationship_between_ordinary_named_regions():
    from vectoreyes.region_addressing import assign_cutout
    from vectoreyes.drawing_actions import AssignCutoutAction
    drawing = Drawing('d', Canvas(10,10), (
        region(name='badge'), region('r2',name='badge-counter',d='M2 2 L4 2 L4 4 L2 4 Z')))
    updated = assign_cutout(drawing, AssignCutoutAction(op='assign_cutout',source='r2',targets=('r1',)))
    assert updated.region('r2').geometry == drawing.region('r2').geometry
    assert updated.region('r1').geometry == drawing.region('r1').geometry
    assert updated.region('r2').cutout_targets == ('r1',)
    rendered = render_drawing(updated)
    xml = ET.fromstring(rendered.svg)
    target = xml.find(".//*[@id='r1']")
    path = target if target.tag.endswith('path') else target.find('.//{http://www.w3.org/2000/svg}path')
    from vectoreyes.skia_geometry import SkPath
    assert SkPath.from_svg_d(path.get('d')).area == pytest.approx(60)
    assert rendered.report['relationships'] == [dict(type='cutout',source='r2',targets=['r1'])]
    assert xml.find('{http://www.w3.org/2000/svg}metadata') is not None


def test_cutout_declaration_by_name_replays_and_can_be_removed():
    evidence, paint, base = fixture()
    # Self subtraction and unresolved names must fail without publishing.
    with pytest.raises(ValueError, match='itself|overlap|differ'):
        submit(evidence,paint,base,[dict(op='assign_cutout',source='r1',targets=['r1'])])
    recipe, output = submit(evidence,paint,base,[
        dict(op='set_region_name',target='r1',name='body'),
        dict(op='assign_cutout',source={'name':'body'},targets=[])])
    assert output.drawing.region('r1').cutout_targets == ()


def test_name_schema_accepts_addresses_in_lists_and_nested_band_spans():
    plan = DrawingPlan.model_validate(dict(version='vectoreyes.plan.v1',drawing_id='d',base_version='v0',ops=[
        dict(op='stitch',targets=[{'name':'orange-band'},'r2']),
        dict(op='set_band', first=dict(region={'name':'orange-band'},between=['r1.p0.c0','r1.p0.c1']),
             second=dict(region='r2',between=['r2.p0.c0','r2.p0.c1']))]))
    assert plan.ops[0].targets[0].name == 'orange-band'
    with pytest.raises(ValidationError):
        DrawingPlan.model_validate(dict(version='vectoreyes.plan.v1',drawing_id='d',base_version='v0',
            ops=[dict(op='set_region_name',target='r1',name='Orange Band')]))


def test_validate_plan_handles_names_and_repeated_renames():
    from vectoreyes.drawing_plan import validate_plan
    drawing=Drawing('d',Canvas(10,10),(region(),))
    plan=DrawingPlan.model_validate(dict(version='vectoreyes.plan.v1',drawing_id='d',base_version='v0',ops=[
        dict(op='set_region_name',target='r1',name='body'),
        dict(op='set_region_name',target={'name':'body'},name='orange-band'),
        dict(op='set_fill',target={'name':'orange-band'},fill=dict(type='flat',color='#ffffff'))]))
    validate_plan(plan,drawing)


def test_cutout_relationship_survives_replay_and_target_split():
    from tests._working_evidence import capture_working_geometry_evidence
    from vectoreyes.refinement.assembly_binding import working_regions_from_evidence
    from vectoreyes.fit import Shape
    evidence,paint,base=fixture()
    first=working_regions_from_evidence(evidence)[0]
    second=replace(first,id=2,drawing_id='r2',source_regions=('r2',),z=1,
        current=Shape('rect',dict(x=2,y=2,w=1,h=1)))
    evidence=capture_working_geometry_evidence(kind='source',trace=evidence.trace,
        geometry_source=evidence.geometry_source,trace_roots=(first,second))
    base=replace(base,evidence_revision=evidence.revision)
    recipe,execution=submit(evidence,paint,base,[dict(op='set_region_name',target='r1',name='body'),
        dict(op='set_region_name',target='r2',name='counter'),
        dict(op='assign_cutout',source={'name':'counter'},targets=[{'name':'body'}])])
    assert execution.drawing.region('r2').cutout_targets==('r1',)
    assert recipe.operations[-1].action.source=='r2'
    recipe,split=submit(evidence,paint,recipe,[dict(op='split',target={'name':'body'},
        divider=dict(type='line',points=[[3,0],[3,8]]))])
    assert split.drawing.region('r1').name=='body'
    assert split.drawing.region('r1').is_branch
    assert split.drawing.region('r2').cutout_targets==('r1',)
    assert replay.reproduce_effective_recipe(drawing=split.drawing,evidence=evidence,paint=paint,
        recipe=recipe,replay=split.replay)==split.drawing


def test_svg_grouping_preserves_interleaved_paint_order_and_unique_ids():
    import resvg_py
    a=replace(region('r1'),z=0)
    b=replace(region('r2'),z=2,fill=FlatFill('#aa0000'))
    c=replace(region('r3'),z=1,fill=FlatFill('#00aa00'))
    group=DrawingRegion('body',children=(a,b))
    drawing=Drawing('d',Canvas(10,10),(group,c))
    named=replace(drawing,regions=(replace(group,name='body'),c))
    before=render_drawing(drawing).svg
    after=render_drawing(named).svg
    assert resvg_py.svg_to_bytes(svg_string=before)==resvg_py.svg_to_bytes(svg_string=after)
    xml=ET.fromstring(after)
    ids=[node.get('id') for node in xml.iter() if node.get('id')]
    assert len(ids)==len(set(ids))


def test_mcp_names_survive_versioning_maps_and_plain_export():
    from types import SimpleNamespace
    from tests.test_mcp_server import _png_data_uri
    from vectoreyes.mcp_server import trace_drawing, refine_drawing, get_drawing_artifact, TraceDrawingOptions, ImageRef
    ctx=SimpleNamespace(session=object())
    traced=trace_drawing(ImageRef(data_uri=_png_data_uri()),TraceDrawingOptions(),ctx).structured_content
    drawing_id=traced['drawing_id']; identity=traced['report']['targets'][0]['id']
    output=refine_drawing(dict(version='vectoreyes.plan.v1',drawing_id=drawing_id,base_version=traced['version'],ops=[
        dict(op='set_region_name',target=identity,name='badge'),
        dict(op='set_fill',target={'name':'badge'},fill=dict(type='flat',color='#00aa00'))]),ctx).structured_content
    assert output['report']['targets'][0]['name']=='badge'
    svg=get_drawing_artifact(drawing_id,output['version'],'svg',ctx).structured_content['svg']
    plain=get_drawing_artifact(drawing_id,output['version'],'svg',ctx,editable_svg=False).structured_content['svg']
    assert 'inkscape:label="badge"' in svg and 'inkscape' not in plain
    labels=get_drawing_artifact(drawing_id,output['version'],'labeled_svg',ctx).structured_content['svg']
    assert 'badge' in labels
    assert traced['report']['targets'][0].get('name') is None


def test_cutout_cycles_and_parent_self_subtraction_are_rejected_at_ingress():
    a=region('r1',cutout_targets=('r2',))
    b=region('r2',cutout_targets=('r1',))
    with pytest.raises(ValueError,match='nested|cycle'):
        Drawing('d',Canvas(10,10),(a,b))
    parent=DrawingRegion('body',children=(region('r1'),region('r2')),cutout_targets=('r1',))
    with pytest.raises(ValueError,match='itself|overlap'):
        Drawing('d',Canvas(10,10),(parent,))


def test_explicit_cutouts_ignore_paint_color_and_order_and_support_mask_export():
    from vectoreyes.output_settings import DrawingOutputSettings
    base=region('r1',name='body',z=2)
    hole=region('r2',name='counter',d='M2 2 L4 2 L4 4 L2 4 Z',z=0,cutout_targets=('r1',))
    drawing=Drawing('d',Canvas(10,10),(base,hole))
    for mode in ('subtract','off','mask'):
        output=render_drawing(replace(drawing,output_settings=DrawingOutputSettings(background_cutouts=mode)))
        xml=ET.fromstring(output.svg)
        target=xml.find(".//*[@id='r1']")
        assert xml.find(".//*[@id='r2']") is None
        assert bool(target.get('mask'))==(mode=='mask')
        assert output.report['targets'][1]['cutout'] is True


def test_report_label_uses_name_and_keeps_exact_id():
    import subprocess
    from pathlib import Path
    script=Path('src/vectoreyes/assets/evaluation/report.js').resolve()
    check='''const fs=require('fs'),vm=require('vm');
const source=fs.readFileSync(process.argv[1],'utf8').split('\\n').find(x=>x.startsWith('function labelForRegion('));
const context={versions:{v0:{inspection:{regions:[{id:'r1',name:'orange-band'}],groups:[]}}},current:()=> 'v0'};
vm.createContext(context);vm.runInContext(source,context);
if(context.labelForRegion('r1')!=='orange-band · r1')throw Error('name missing');
if(context.labelForRegion('r2')!=='r2')throw Error('unknown ID changed');'''
    subprocess.run(['node','-e',check,str(script)],check=True,capture_output=True,text=True)


def test_disposable_numeric_adapter_never_silently_ignores_metadata_actions():
    from vectoreyes.drawing_refine import refine
    from vectoreyes.refinement.assembly_binding import working_regions_from_evidence
    evidence,_,_=fixture()
    plan=DrawingPlan.model_validate(dict(version='vectoreyes.plan.v1',drawing_id='drw_test',base_version='v0',
        ops=[dict(op='set_region_name',target='r1',name='body')]))
    with pytest.raises(ValueError,match='DrawingEngine'):
        refine(evidence.trace,working_regions_from_evidence(evidence),plan)


def test_dedicated_naming_tool_is_atomic_and_uses_ordinary_revisions():
    from types import SimpleNamespace
    from tests.test_mcp_server import _png_data_uri
    import vectoreyes.mcp_server as server
    assert hasattr(server,'name_regions'), 'dedicated region naming tool is missing'
    ctx=SimpleNamespace(session=object())
    traced=server.trace_drawing(server.ImageRef(data_uri=_png_data_uri()),server.TraceDrawingOptions(),ctx).structured_content
    identity=traced['report']['targets'][0]['id']; drawing_id=traced['drawing_id']
    before=server._DRAWINGS.get_version(drawing_id,traced['version']).drawing
    named=server.name_regions(drawing_id,traced['version'],[dict(target=identity,name='badge')],ctx=ctx).structured_content
    assert named['parent_version']==traced['version']
    assert named['report']['targets'][0]['name']=='badge'
    after=server._DRAWINGS.get_version(drawing_id,named['version']).drawing
    assert after.region(identity).geometry==before.region(identity).geometry
    assert after.region(identity).fill==before.region(identity).fill
    manifest=server._DRAWINGS.get_manifest(drawing_id)
    with pytest.raises(Exception,match='absent|unknown|missing|not found'):
        server.name_regions(drawing_id,named['version'],[dict(target=identity,name='new-badge'),
            dict(target={'name':'missing'},name='other')],ctx=ctx)
    assert server._DRAWINGS.get_manifest(drawing_id)==manifest
    assert after.region(identity).name=='badge'
