"""
Tests for VaccineInfoAgent
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sub_agents.vaccine_info_agent import VaccineInfoAgent, VACCINE_KB
from memory import InMemorySessionService, MemoryBank


@pytest.fixture
def agent(config, memory_bank):
    return VaccineInfoAgent(config, memory_bank=memory_bank)


@pytest.fixture
def session():
    return InMemorySessionService().create_session("test_user")


class TestVaccineInfoAgentInit:
    def test_initialization(self, config):
        agent = VaccineInfoAgent(config)
        assert agent.name == "vaccine_info_agent"
        assert agent.kb == VACCINE_KB

    def test_initialization_with_memory_bank(self, config, memory_bank):
        agent = VaccineInfoAgent(config, memory_bank=memory_bank)
        assert agent.memory_bank is memory_bank


class TestKnowledgeBaseRetrieval:
    def test_retrieve_side_effects(self, agent):
        result = agent._retrieve_kb_context("What are the side effects?")
        assert len(result) > 0
        assert any("side effect" in r.lower() for r in result)

    def test_retrieve_safety(self, agent):
        result = agent._retrieve_kb_context("Are vaccines safe?")
        assert len(result) > 0
        assert any("safe" in r.lower() for r in result)

    def test_retrieve_eligibility(self, agent):
        result = agent._retrieve_kb_context("Am I eligible?")
        assert len(result) > 0
        assert any("eligible" in r.lower() for r in result)

    def test_retrieve_default_overview(self, agent):
        result = agent._retrieve_kb_context("random question")
        assert len(result) > 0
        assert VACCINE_KB["overview"] in result


class TestContextCompaction:
    def test_compaction_under_limit(self, agent):
        session = {"history": [{"text": f"msg {i}"} for i in range(5)]}
        agent._compact_context(session)
        assert len(session["history"]) == 5

    def test_compaction_over_limit(self, agent):
        session = {"history": [{"text": f"msg {i}"} for i in range(20)]}
        agent._compact_context(session)
        assert len(session["history"]) == agent.max_history_length


class TestPromptBuilding:
    def test_build_prompt_contains_kb(self, agent):
        kb_context = [VACCINE_KB["overview"]]
        prompt = agent._build_prompt("What is a vaccine?", kb_context)
        assert "KNOWLEDGE BASE" in prompt
        assert VACCINE_KB["overview"] in prompt

    def test_build_prompt_contains_question(self, agent):
        prompt = agent._build_prompt("Test question?", [])
        assert "Test question?" in prompt


class TestTopicInference:
    def test_infer_side_effects(self, agent):
        assert agent._infer_topic("side effects of vaccines") == "side_effects"

    def test_infer_safety(self, agent):
        assert agent._infer_topic("are vaccines safe") == "safety"

    def test_infer_general(self, agent):
        assert agent._infer_topic("tell me something") == "general"


class TestOnEvent:
    @pytest.mark.asyncio
    async def test_on_event_returns_response(self, agent, session):
        event = MockEvent({"text": "What is a vaccine?"})
        ctx = MockCtx(session)

        response = await agent.on_event(event, ctx)

        assert hasattr(response, 'text')
        assert len(response.text) > 0

    @pytest.mark.asyncio
    async def test_on_event_updates_history(self, agent, session):
        initial_len = len(session["history"])
        event = MockEvent({"text": "Tell me about vaccines"})
        ctx = MockCtx(session)

        await agent.on_event(event, ctx)

        assert len(session["history"]) > initial_len

    @pytest.mark.asyncio
    async def test_on_event_tracks_metrics(self, agent, session):
        event = MockEvent({"text": "Test question"})
        ctx = MockCtx(session)

        await agent.on_event(event, ctx)

        assert ctx.metrics.counters.get("vaccine_info_queries:None", 0) > 0


class TestMemoryBankIntegration:
    @pytest.mark.asyncio
    async def test_saves_to_memory_bank(self, config, memory_bank, session):
        agent = VaccineInfoAgent(config, memory_bank=memory_bank)
        event = MockEvent({"text": "What are vaccines?"})
        ctx = MockCtx(session)

        await agent.on_event(event, ctx)

        memories = memory_bank.get(session["user_id"])
        assert len(memories) > 0

    def test_get_preferred_language_default(self, agent, session):
        lang = agent._get_preferred_language(session)
        assert lang == "en"

    def test_get_preferred_language_from_session(self, agent, session):
        session["lang"] = "es"
        lang = agent._get_preferred_language(session)
        assert lang == "es"