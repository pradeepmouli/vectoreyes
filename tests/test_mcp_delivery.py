import base64
import hashlib
import json
import asyncio
from types import SimpleNamespace

import pytest
from mcp.types import CallToolResult, TextContent

import vectoreyes.mcp_server as server
from tests.test_mcp_server import _png_data_uri


def trace():
    return server.trace_drawing(image_data=server.DirectImageRef(data_uri=_png_data_uri()),
        options=server.TraceDrawingOptions(),ctx=SimpleNamespace(session=object()))


def test_drawing_response_does_not_duplicate_full_json_as_text():
    result=trace()
    text=next(block.text for block in result.content if block.type=='text')
    assert len(text.encode())<1024
    assert result.structured_content['report']['targets']


def test_oversized_reports_are_explicitly_deferred_before_publication():
    result=trace().structured_content
    result['report']['targets'][0]['diagnostics']['large']='x'*(10*1024*1024)
    response=server._validated_drawing_result(server.TraceDrawingOutput,result,b'png')
    assert response.structured_content['report'] is None
    assert response.structured_content['delivery']['reports_deferred'] is True
    assert all(output['report'] is None for output in response.structured_content['outputs'])
    assert len(response.model_dump_json().encode())<8*1024*1024


def test_artifact_pages_reassemble_exact_unicode_payload():
    original=CallToolResult(content=[TextContent(type='text',text='SVG available in structuredContent')],
        structuredContent={'mime_type':'image/svg+xml','svg':'<svg>'+'🍊<&'*(1024*1024)+'</svg>'})
    raw=original.model_dump_json(by_alias=True,exclude_none=True).encode()
    chunks=[]; offset=0
    while True:
        response=server._artifact_page(original,offset=offset)
        chunk=response.structured_content['chunk']
        assert len(response.model_dump_json().encode())<8*1024*1024
        assert chunk['offset']==offset
        assert chunk['sha256']==hashlib.sha256(raw).hexdigest()
        chunks.append(base64.b64decode(chunk['data']))
        if chunk['next_offset'] is None: break
        offset=chunk['next_offset']
    assert b''.join(chunks)==raw
    with pytest.raises(Exception,match='offset'):
        server._artifact_page(original,offset=-1)


def test_full_report_can_be_requested_without_opening_widget():
    result=trace().structured_content
    report=server.get_drawing_artifact(result['drawing_id'],result['version'],artifact='report',
        ctx=SimpleNamespace(session=object()))
    assert report.structured_content['report']==result['report']
    assert not report.meta


def test_evaluation_client_reassembles_and_verifies_artifact_pages():
    from vectoreyes.evaluation.client import read_artifact
    original=CallToolResult(content=[],structuredContent={'mime_type':'image/svg+xml','svg':'x'*(7*1024*1024)})
    class Client:
        async def call_tool(self,name,arguments):
            return server._artifact_page(original,offset=arguments.get('offset',0))
    result=asyncio.run(read_artifact(Client(),{'drawing_id':'test','version':'v0','artifact':'svg'}))
    assert result.structured_content==original.structured_content


def test_large_artifact_roundtrips_through_actual_mcp_schema(monkeypatch):
    from fastmcp import Client
    from vectoreyes.evaluation.client import read_artifact
    svg='<svg>'+'x'*(11*1024*1024)+'</svg>'
    original=CallToolResult(content=[],structuredContent={'mime_type':'image/svg+xml','svg':svg})
    monkeypatch.setattr(server,'_get_drawing_artifact',lambda *args,**kwargs:original)
    async def run():
        async with Client(server.mcp) as client:
            return await read_artifact(client,{'drawing_id':'test','version':'v0','artifact':'svg'})
    result=asyncio.run(run())
    assert result.structured_content['svg']==svg


def test_inline_review_image_is_bounded_without_changing_settings():
    import io
    import numpy as np
    from PIL import Image
    result=trace().structured_content
    buffer=io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0,256,(768,768,3),dtype=np.uint8)).save(buffer,format='PNG')
    assert len(buffer.getvalue())>1024*1024
    response=server._validated_drawing_result(server.TraceDrawingOutput,result,buffer.getvalue())
    image=next(block for block in response.content if block.type=='image')
    assert len(base64.b64decode(image.data))<=1024*1024
    assert response.structured_content['delivery']['review_panel_resized']
    assert response.structured_content['trace_options']==result['trace_options']


def test_large_trace_report_is_delivered_then_retrieved_through_mcp(monkeypatch):
    from fastmcp import Client
    from vectoreyes.evaluation.client import read_artifact
    summary=server.canonical_drawing_summary
    diagnostic='large-report-evidence-'*(512*1024)
    def large_summary(*args,**kwargs):
        report=summary(*args,**kwargs)
        report['targets'][0]['diagnostics']['large']=diagnostic
        return report
    monkeypatch.setattr(server,'canonical_drawing_summary',large_summary)
    async def run():
        async with Client(server.mcp) as client:
            traced=await client.call_tool('vectoreyes_trace_drawing',{
                'image_data':{'data_uri':_png_data_uri()},'options':{'refine':'none'}})
            result=traced.structured_content
            assert result['report'] is None
            assert result['delivery']['reports_deferred']
            report=await read_artifact(client,{'drawing_id':result['drawing_id'],
                'version':result['version'],'artifact':'report'})
            assert report.structured_content['report']['targets'][0]['diagnostics']['large']==diagnostic
    asyncio.run(run())
