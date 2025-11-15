import pytest
import asyncio

from sub_agents.vaccine_info_agent import VaccineInfoAgent
from memory import InMemorySessionService

# Minimal ADK event stub
class FakeEvent:
    def __init__(self, payload):
        self.payload = payload
        self.resume = False


# Minimal ADK context stub
class FakeCtx:
    def __init__(self, session):
        self.session = session
        self.metrics = FakeMetrics()

    async def call_tool(self, name, args):
        raise NotImplementedError()

    async def call_model(self, prompt):
        return "mocked LLM response"


class FakeMetrics:
    def __init__(self):
        self.counters = {}

    def increment(self, name):
        self.counters[name] = self.counters.get(name, 0) + 1


@pytest.mark.asyncio
async def test_vaccine_info_basic_overview():
    config = {"model": "gpt-5-mini"}
    agent = VaccineInfoAgent(config)
    session = InMemorySessionService().create_session("test_user")

    # simulate user asking
    event = FakeEvent({"text": "What is a vaccine?"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    assert "vaccines" in response.text.lower() or "immune" in response.text.lower()
    assert len(session["history"]) >= 1


@pytest.mark.asyncio
async def test_vaccine_info_side_effects():
    config = {"model": "gpt-5-mini"}
    agent = VaccineInfoAgent(config)
    session = InMemorySessionService().create_session("test_user")

    event = FakeEvent({"text": "What are the common side effects?"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)
    txt = response.text.lower()

    assert "side" in txt or "fever" in txt or "aches" in txt


@pytest.mark.asyncio
async def test_vaccine_info_context_compaction():
    config = {"model": "gpt-5-mini"}
    agent = VaccineInfoAgent(config)
    session = InMemorySessionService().create_session("test_user")

    # artificially extend history
    for i in range(20):
        session["history"].append({"role": "user", "text": f"msg {i}"})

    event = FakeEvent({"text": "Tell me about vaccines"})
    ctx = FakeCtx(session)

    response = await agent.on_event(event, ctx)

    # After compaction, history should have assistant message added
    assert len(session["history"]) >= 9  # last 8 + new response


@pytest.mark.asyncio
async def test_vaccine_info_metrics_increment():
    config = {"model": "gpt-5-mini"}
    agent = VaccineInfoAgent(config)
    session = InMemorySessionService().create_session("user_metrics")

    event = FakeEvent({"text": "Give me vaccine safety info"})
    ctx = FakeCtx(session)

    await agent.on_event(event, ctx)

    assert ctx.metrics.counters.get("vaccine_info_queries", 0) == 1
