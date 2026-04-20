import json
from typing import Any, cast

import pytest
from anthropic import AsyncAnthropic

from agent_forge.agents.editor import Editor
from agent_forge.agents.hashtag_specialist import HashtagSpecialist
from agent_forge.agents.image_prompter import ImagePrompter
from agent_forge.observability.tracer import InMemoryTracer


class _FakeTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeUsage:
    input_tokens = 10
    output_tokens = 5
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _FakeMessage:
    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage()
        self.stop_reason = "end_turn"


class _FakeMessages:
    def __init__(self, text: str) -> None:
        self._text = text
        self.call_kwargs: dict[str, Any] | None = None

    async def create(self, **kwargs: Any) -> _FakeMessage:
        self.call_kwargs = kwargs
        return _FakeMessage(self._text)


class _FakeClient:
    def __init__(self, text: str) -> None:
        self.messages = _FakeMessages(text)


def test_editor_wires_specialist_model_and_prompt() -> None:
    agent = Editor(client=cast(AsyncAnthropic, object()), tracer=InMemoryTracer())

    assert agent.name == "editor"
    assert agent.model == "claude-sonnet-4-6"
    assert agent.thinking is None
    assert agent.output_schema is None
    assert "Editor" in agent.system_prompt
    assert "refine" in agent.system_prompt.lower()


def test_hashtag_specialist_uses_fast_model_with_schema() -> None:
    agent = HashtagSpecialist(client=cast(AsyncAnthropic, object()), tracer=InMemoryTracer())

    assert agent.name == "hashtag_specialist"
    assert agent.model == "claude-haiku-4-5-20251001"
    assert agent.output_schema is not None
    assert "hashtags" in agent.output_schema["properties"]


async def test_hashtag_specialist_pick_parses_valid_response() -> None:
    client = _FakeClient(json.dumps({"hashtags": ["#AI", "#LLMs", "#Engineering"]}))
    agent = HashtagSpecialist(client=cast(AsyncAnthropic, client), tracer=InMemoryTracer())

    tags = await agent.pick(post="some post", run_id="run-1")

    assert tags == ["#AI", "#LLMs", "#Engineering"]


@pytest.mark.parametrize("tag_count", [2, 6])
async def test_hashtag_specialist_rejects_wrong_count(tag_count: int) -> None:
    tags = [f"#Tag{i}" for i in range(tag_count)]
    client = _FakeClient(json.dumps({"hashtags": tags}))
    agent = HashtagSpecialist(client=cast(AsyncAnthropic, client), tracer=InMemoryTracer())

    with pytest.raises(ValueError, match="Expected 3-5 hashtags"):
        await agent.pick(post="some post", run_id="run-1")


def test_image_prompter_uses_fast_model_with_schema() -> None:
    agent = ImagePrompter(client=cast(AsyncAnthropic, object()), tracer=InMemoryTracer())

    assert agent.name == "image_prompter"
    assert agent.model == "claude-haiku-4-5-20251001"
    assert agent.output_schema is not None
    slide_schema = agent.output_schema["properties"]["slides"]["items"]
    assert "title" in slide_schema["properties"]
    assert "image_prompt" in slide_schema["properties"]


async def test_image_prompter_design_parses_valid_response() -> None:
    payload = {
        "slides": [
            {"title": "Hook", "image_prompt": "close-up shot"},
            {"title": "Insight", "image_prompt": "split-screen diagram"},
            {"title": "Takeaway", "image_prompt": "minimalist card"},
        ]
    }
    client = _FakeClient(json.dumps(payload))
    agent = ImagePrompter(client=cast(AsyncAnthropic, client), tracer=InMemoryTracer())

    slides = await agent.design(post="some post", run_id="run-1")

    assert len(slides) == 3
    assert slides[0].title == "Hook"
    assert slides[2].image_prompt == "minimalist card"


@pytest.mark.parametrize("slide_count", [2, 5])
async def test_image_prompter_rejects_wrong_slide_count(slide_count: int) -> None:
    payload = {
        "slides": [{"title": f"T{i}", "image_prompt": f"prompt {i}"} for i in range(slide_count)]
    }
    client = _FakeClient(json.dumps(payload))
    agent = ImagePrompter(client=cast(AsyncAnthropic, client), tracer=InMemoryTracer())

    with pytest.raises(ValueError, match="Expected 3-4 slides"):
        await agent.design(post="some post", run_id="run-1")
