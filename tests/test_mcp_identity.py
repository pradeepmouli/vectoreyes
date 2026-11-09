"""The server advertises the same packaged radish used by its widget."""

import asyncio
import base64
import os
import sys
from importlib.resources import files

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def test_initialize_advertises_the_packaged_vectoreyes_icon():
    async def initialize():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "vectoreyes.mcp_server"],
            env={**os.environ, "VECTOREYES_MCP_TRANSPORT": "stdio"},
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                return await session.initialize()

    result = asyncio.run(initialize())
    assert result.server_info.name == "vectoreyes"
    assert result.server_info.icons, "MCP initialization must advertise the vectorEyes eye"
    icon = result.server_info.icons[0]
    assert icon.mime_type == "image/svg+xml"
    assert icon.sizes == ["any"]
    prefix, payload = icon.src.split(",", 1)
    assert prefix == "data:image/svg+xml;base64"
    assert base64.b64decode(payload) == files("vectoreyes").joinpath("assets/vectoreyes-icon.svg").read_bytes()
