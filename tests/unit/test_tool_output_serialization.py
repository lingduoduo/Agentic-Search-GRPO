"""Tool responses must be JSON, because two consumers already assume they are.

``tool_agent_runner._extract_tool_calls_and_docs`` calls json.loads on the
response to recover structure, and ``_fit_json_array`` trims a JSON array by
whole items. A Python repr defeats both silently. See #596.
"""

import asyncio
import json

from src.agents.tool.tool_calling import _truncate_tool_text
from src.internal.tools.base import FunctionTool


def _execute(fn, **arguments):
    tool = FunctionTool(fn=fn, name=getattr(fn, "__name__", "t"))

    async def run():
        instance_id = await tool.create()
        try:
            return await tool.execute(instance_id, arguments)
        finally:
            await tool.release(instance_id)

    return asyncio.run(run())


async def _dict_tool(symbol: str) -> dict:
    return {"symbol": symbol.upper(), "price": 123.45, "previous_close": None}


async def _list_tool(count: int) -> list:
    return [
        {
            "title": f"Result {i}",
            "url": f"https://example.org/{i}",
            "snippet": "x" * 260,
        }
        for i in range(count)
    ]


async def _str_tool(text: str) -> str:
    return text


async def _unserializable_tool() -> object:
    return object()


async def _unicode_tool() -> dict:
    return {"city": "Köln", "note": "naïve café"}


def test_a_dict_result_round_trips_through_json():
    """This is exactly the recovery the source-card builder attempts."""
    response, raw, _ = _execute(_dict_tool, symbol="aapl")

    assert json.loads(response) == raw
    assert json.loads(response)["symbol"] == "AAPL"


def test_a_string_result_is_passed_through_unchanged():
    """json.dumps on a str would add quotes and change every text tool."""
    response, raw, _ = _execute(_str_tool, text='plain text with "quotes"')

    assert response == 'plain text with "quotes"'
    assert response == raw


def test_a_long_list_result_is_trimmed_by_whole_items():
    """_fit_json_array only engages for real JSON; a repr falls to slicing."""
    response, _, _ = _execute(_list_tool, count=12)
    assert len(response) > 2048

    truncated = _truncate_tool_text(response, 2048, "left")

    assert "results shown" in truncated
    assert "omitted for length" in truncated
    assert "...(truncated)" not in truncated
    body = truncated.split("\n...")[0]
    assert isinstance(json.loads(body), list)


def test_non_ascii_is_not_escaped():
    """ensure_ascii=False: \\uXXXX inflates _fit_json_array's size accounting."""
    response, _, _ = _execute(_unicode_tool)

    assert "Köln" in response
    assert "\\u" not in response


def test_an_unserializable_result_degrades_instead_of_raising():
    response, _, _ = _execute(_unserializable_tool)

    assert isinstance(response, str)
    assert response
