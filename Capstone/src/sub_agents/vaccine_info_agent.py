"""
Vaccine Information Agent

Provides clear, factual vaccine information using a curated knowledge base
and LLM-powered responses. Handles education questions about eligibility,
side-effects, safety, and vaccine types.

Features:
- Knowledge base retrieval with semantic keyword matching
- Context compaction for long conversations
- Memory bank integration for user preferences
- Structured logging and metrics
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from google.genai import types
from pydantic import PrivateAttr


# Use the try-except block to handle testing environment
try:
    from google.adk import Agent
    from google.adk.events import EventActions
except ImportError:
    # Define simple mock classes for testing
    class Agent:
        def __init__(self, *args, **kwargs):
            pass

    class EventActions:
        def __init__(self, **kwargs):
            pass

logger = logging.getLogger(__name__)

MAX_HISTORY_LENGTH = 8
VACCINE_KB = {
    "overview": (
        "Vaccines help your immune system recognize and respond quickly to "
        "dangerous pathogens, reducing your risk of severe illness."
    ),
    "side_effects": (
        "Common side effects include: sore arm, mild fever, fatigue, muscle aches. "
        "Serious reactions are extremely rare, but seek medical help if you "
        "experience difficulty breathing, chest pain, or severe swelling."
    ),
    "eligibility": (
        "Eligibility may depend on age, health conditions, local guidelines, "
        "and vaccine supply. Most adults and children above 6 months are generally "
        "eligible for common vaccines."
    ),
    "safety": (
        "Vaccines undergo extensive testing for safety and effectiveness. "
        "They continue to be monitored by global health organizations."
    ),
    "types": (
        "Common vaccine types include mRNA vaccines, viral vector vaccines, "
        "and inactivated or weakened virus vaccines. Each type works differently "
        "but all help train your immune system."
    ),
    "effectiveness": (
        "Vaccines are highly effective at preventing severe disease, hospitalization, "
        "and death. Effectiveness may vary by vaccine type and variant."
    ),
    "boosters": (
        "Booster doses help maintain immunity over time. Your healthcare provider "
        "can advise on the recommended booster schedule for your situation."
    ),
    "myths": (
        "Common myths: Vaccines do NOT cause autism, do NOT alter DNA, and do NOT "
        "contain microchips. These claims have been thoroughly debunked by scientific research."
    ),
}


class VaccineInfoAgent(Agent):
    """
    Education-focused agent that answers vaccine questions using a knowledge base
    and LLM enhancement for natural language responses.
    """

    # Declare private attributes using PrivateAttr
    _memory_bank: Any = PrivateAttr(default=None)
    _kb: Any = PrivateAttr(default=None)
    _max_history_length: Any = PrivateAttr(default=None)
    _config: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, config: Dict[str, Any], memory_bank=None):

        super().__init__(
            name="vaccine_info_agent",
            model=config.get("model", "gemini-2.0-flash"),
            description=(
                "Provides accurate, accessible vaccine information. "
                "Answers questions about safety, side effects, eligibility, and effectiveness."
            ),
            instruction=(
                "You are a helpful vaccine education assistant. Use the knowledge base "
                "provided in context to answer questions accurately. Never provide medical "
                "diagnosis. If asked about personal medical advice, recommend consulting "
                "a healthcare provider. Keep responses clear, empathetic, and factual."
            ),
        )
        # Use private attributes to avoid Pydantic field conflicts
        object.__setattr__(self, '_config', config)
        object.__setattr__(self, '_memory_bank', memory_bank)
        object.__setattr__(self, '_kb', VACCINE_KB)
        object.__setattr__(self, '_max_history_length', config.get("max_history_length", MAX_HISTORY_LENGTH))

    @property
    def config(self):
        return self._config

    @property
    def memory_bank(self):
        return self._memory_bank

    @property
    def kb(self):
        return self._kb

    @property
    def max_history_length(self):
        return self._max_history_length

    async def on_event(self, event, ctx):
        """Handle vaccine information requests."""
        session = ctx.session
        payload = event.payload or {}
        user_query = payload.get("text", "").strip()

        logger.info(
            "VaccineInfoAgent processing query",
            extra={
                "agent": self.name,
                "query_length": len(user_query),
                "session_id": session.get("user_id", "unknown")[:8],
            }
        )

        if hasattr(ctx, "metrics") and ctx.metrics:
            ctx.metrics.increment("vaccine_info_queries")

        preferred_lang = self._get_preferred_language(session)
        self._compact_context(session)
        kb_context = self._retrieve_kb_context(user_query)
        enhanced_prompt = self._build_prompt(user_query, kb_context, preferred_lang)

        try:
            response_text = await ctx.call_model(enhanced_prompt)
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            response_text = (
                "I apologize, but I'm having trouble accessing information right now. "
                "Please try again in a moment."
            )
            if hasattr(ctx, "metrics") and ctx.metrics:
                ctx.metrics.increment("vaccine_info_errors")

        session["history"].append({
            "role": "user",
            "text": user_query,
            "timestamp": datetime.utcnow().isoformat()
        })
        session["history"].append({
            "role": "assistant",
            "text": response_text,
            "timestamp": datetime.utcnow().isoformat()
        })

        if self._memory_bank:
            self._update_memory(session, user_query, kb_context)

        logger.info("VaccineInfoAgent response generated", extra={"response_length": len(response_text)})

        return EventActions(state_delta={"message": response_text})

    def _compact_context(self, session: Dict[str, Any]):
        """Keep only the last N messages to avoid token overflow."""
        history = session.get("history", [])
        if len(history) > self._max_history_length:
            session["history"] = history[-self._max_history_length:]
            logger.info(
                "Context compacted",
                extra={"original_length": len(history), "compacted_length": len(session["history"])}
            )

    def _retrieve_kb_context(self, query: str) -> List[str]:
        """Retrieve relevant KB entries using keyword matching."""
        query_lower = query.lower()
        relevant = []

        keywords_map = {
            "side effect": ["side_effects"],
            "safe": ["safety"],
            "effective": ["effectiveness"],
            "eligible": ["eligibility"],
            "type": ["types"],
            "booster": ["boosters"],
            "myth": ["myths"],
            "work": ["overview"],
            "what is": ["overview"],
            "how": ["overview"],
        }

        for keyword, kb_keys in keywords_map.items():
            if keyword in query_lower:
                for kb_key in kb_keys:
                    if kb_key in self._kb:
                        relevant.append(self._kb[kb_key])

        if not relevant:
            relevant.append(self._kb["overview"])

        return relevant

    def _build_prompt(self, user_query: str, kb_context: List[str], language: str = "en") -> str:
        """Build enhanced prompt with knowledge base context."""
        kb_text = "\n\n".join(kb_context)

        return f"""You are a vaccine education assistant. Use the following trusted information to answer the user's question:

KNOWLEDGE BASE:
{kb_text}

USER QUESTION: {user_query}

INSTRUCTIONS:
- Provide a clear, factual answer based on the knowledge base
- If the question is outside the knowledge base scope, acknowledge this and suggest consulting a healthcare provider
- Do not provide medical diagnosis or personal medical advice
- Keep the response empathetic, accessible, and around 2-3 sentences unless more detail is requested
- Respond in {language} language if not English
"""

    def _get_preferred_language(self, session: Dict[str, Any]) -> str:
        """Retrieve user's preferred language from memory bank or session."""
        if self._memory_bank:
            user_id = session.get("user_id")
            if user_id:
                memories = self._memory_bank.get(user_id)
                for mem in memories:
                    if "preferred_language" in mem:
                        return mem["preferred_language"]
        return session.get("lang", "en")

    def _update_memory(self, session: Dict[str, Any], query: str, kb_context: List[str]):
        """Store query metadata in memory bank for analytics."""
        user_id = session.get("user_id")
        if not user_id:
            return

        memory_entry = {
            "event": "vaccine_info_query",
            "query_topic": self._infer_topic(query),
            "timestamp": datetime.utcnow().isoformat(),
            "kb_sections_used": len(kb_context),
        }

        self._memory_bank.save(user_id, memory_entry)

    def _infer_topic(self, query: str) -> str:
        """Infer the main topic from the query."""
        query_lower = query.lower()

        topic_keywords = {
            "side_effects": ["side effect"],
            "safety": ["safe", "safety"],
            "eligibility": ["eligible"],
            "types": ["type"],
            "boosters": ["booster"],
            "myths": ["myth"],
            "effectiveness": ["effective"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in query_lower for kw in keywords):
                return topic

        return "general"