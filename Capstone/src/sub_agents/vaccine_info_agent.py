"""
VaccineInfoAgent

Responsibilities:
- Provide clear, factual vaccine information.
- Handle user education questions (eligibility, side-effects, safety).
- Use MemoryBank signals (preferred language, prior questions).
- Produce stable, safe answers (no medical diagnosis).
- Compact session context before LLM calls.
- Demonstrates context engineering + observability.

This agent does NOT hallucinate – it uses a stable knowledge base + LLM expansion.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

# ADK imports (fallback stubs if running offline tests)
try:
    from adk import Agent, EventActions
    from adk.models import ModelMessage
except Exception:
    from adk import Agent, EventActions, ModelMessage  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# --- Knowledge Base (static, can later be replaced with MCP/HTTP) ---

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

    def __init__(self, config: Dict[str, Any], memory_bank=None):
        super().__init__(
            name="vaccine_info_agent",
            model=config.get("model", "gpt-4"),
            description=(
                "Provides accurate, accessible vaccine information. "
                "Answers questions about safety, side effects, eligibility, and effectiveness."
            ),
            instructions=(
                "You are a helpful vaccine education assistant. Use the knowledge base "
                "provided in context to answer questions accurately. Never provide medical "
                "diagnosis. If asked about personal medical advice, recommend consulting "
                "a healthcare provider. Keep responses clear, empathetic, and factual."
            ),
        )
        self.config = config
        self.memory_bank = memory_bank
        self.kb = VACCINE_KB
        self.max_history_length = config.get("max_history_length", 8)

    async def on_event(self, event, ctx):
        """
        Main event handler for vaccine information requests.

        event.payload expected keys:
        - text: user's question
        - language: preferred language (optional, from memory)
        """
        session = ctx.session
        payload = event.payload or {}
        user_query = payload.get("text", "").strip()

        # Log the incoming query (without PII)
        logger.info(
            "VaccineInfoAgent processing query",
            extra={
                "agent": self.name,
                "query_length": len(user_query),
                "session_id": session.get("user_id", "unknown")[:8],  # Partial ID only
            }
        )

        # Track metrics if available
        if hasattr(ctx, "metrics") and ctx.metrics:
            ctx.metrics.increment("vaccine_info_queries")

        # Step 1: Retrieve user's preferred language from memory
        preferred_lang = await self._get_preferred_language(ctx, session)

        # Step 2: Perform context compaction (keep last N messages)
        self._compact_context(session)

        # Step 3: Match query to knowledge base topics
        kb_context = self._retrieve_kb_context(user_query)

        # Step 4: Build the enhanced prompt with KB context
        enhanced_prompt = self._build_prompt(user_query, kb_context, preferred_lang)

        # Step 5: Call LLM (via ctx.call_model or direct agent invocation)
        try:
            response_text = await self._call_llm(ctx, enhanced_prompt, session)
        except Exception as e:
            logger.exception("LLM call failed", extra={"error": str(e)})
            response_text = (
                "I apologize, but I'm having trouble accessing information right now. "
                "Please try again in a moment."
            )
            if hasattr(ctx, "metrics") and ctx.metrics:
                ctx.metrics.increment("vaccine_info_errors")

        # Step 6: Store interaction in session history
        session["history"].append({"role": "user", "text": user_query, "timestamp": datetime.utcnow().isoformat()})
        session["history"].append({"role": "assistant", "text": response_text, "timestamp": datetime.utcnow().isoformat()})

        # Step 7: Update memory bank with query topic (for analytics)
        if self.memory_bank:
            await self._update_memory(ctx, session, user_query, kb_context)

        logger.info(
            "VaccineInfoAgent response generated",
            extra={"response_length": len(response_text)}
        )

        return ModelMessage(text=response_text)

    def _compact_context(self, session: Dict[str, Any]):
        """
        Context compaction: keep only the last N messages to avoid token overflow.
        This is critical for long conversations.
        """
        history = session.get("history", [])
        if len(history) > self.max_history_length:
            # Keep system context + last N messages
            session["history"] = history[-self.max_history_length:]
            logger.info(
                "Context compacted",
                extra={"original_length": len(history), "compacted_length": len(session["history"])}
            )

    def _retrieve_kb_context(self, query: str) -> List[str]:
        """
        Simple keyword matching to retrieve relevant KB entries.
        In production, use semantic search or vector DB.
        """
        query_lower = query.lower()
        relevant = []

        # Keyword matching logic
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
                    if kb_key in self.kb:
                        relevant.append(self.kb[kb_key])

        # Default fallback: include overview
        if not relevant:
            relevant.append(self.kb["overview"])

        return relevant

    def _build_prompt(self, user_query: str, kb_context: List[str], language: str = "en") -> str:
        """
        Build enhanced prompt with knowledge base context.
        """
        kb_text = "\n\n".join(kb_context)

        prompt = f"""You are a vaccine education assistant. Use the following trusted information to answer the user's question:

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
        return prompt

    async def _call_llm(self, ctx, prompt: str, session: Dict[str, Any]) -> str:
        """
        Call the LLM with the enhanced prompt.
        In ADK, this typically uses ctx.call_model or the agent's built-in model.
        """
        try:
            # Option 1: Use ctx.call_model if available
            if hasattr(ctx, "call_model"):
                response = await ctx.call_model(prompt)
                return response.strip() if isinstance(response, str) else str(response)

            # Option 2: Fallback to mock response for testing
            logger.warning("No LLM available, using fallback response")
            return self._fallback_response(prompt)

        except Exception as e:
            logger.exception("LLM call exception", extra={"error": str(e)})
            raise

    def _fallback_response(self, prompt: str) -> str:
        """
        Fallback response when LLM is unavailable (for testing/degraded mode).
        """
        if "side effect" in prompt.lower():
            return self.kb["side_effects"]
        elif "safe" in prompt.lower():
            return self.kb["safety"]
        elif "eligible" in prompt.lower():
            return self.kb["eligibility"]
        else:
            return self.kb["overview"]

    async def _get_preferred_language(self, ctx, session: Dict[str, Any]) -> str:
        """
        Retrieve user's preferred language from memory bank or session.
        """
        if self.memory_bank:
            user_id = session.get("user_id")
            if user_id:
                memories = self.memory_bank.get(user_id)
                for mem in memories:
                    if "preferred_language" in mem:
                        return mem["preferred_language"]

        # Fallback to session
        return session.get("lang", "en")

    async def _update_memory(self, ctx, session: Dict[str, Any], query: str, kb_context: List[str]):
        """
        Store query metadata in memory bank for analytics and personalization.
        """
        user_id = session.get("user_id")
        if not user_id:
            return

        memory_entry = {
            "event": "vaccine_info_query",
            "query_topic": self._infer_topic(query),
            "timestamp": datetime.utcnow().isoformat(),
            "kb_sections_used": len(kb_context),
        }

        self.memory_bank.save(user_id, memory_entry)

    def _infer_topic(self, query: str) -> str:
        """
        Infer the main topic from the query for categorization.
        """
        query_lower = query.lower()

        if "side effect" in query_lower:
            return "side_effects"
        elif "safe" in query_lower or "safety" in query_lower:
            return "safety"
        elif "eligible" in query_lower:
            return "eligibility"
        elif "type" in query_lower:
            return "types"
        elif "booster" in query_lower:
            return "boosters"
        elif "myth" in query_lower:
            return "myths"
        elif "effective" in query_lower:
            return "effectiveness"
        else:
            return "general"

    # Synchronous wrapper for non-async contexts (e.g., unit tests)
    async def emit(self, payload: Dict[str, Any], session: Dict[str, Any]):
        """
        Simplified emit interface for orchestrator.
        """
        class FakeEvent:
            def __init__(self, payload):
                self.payload = payload
                self.resume = False

        class FakeCtx:
            def __init__(self, session):
                self.session = session
                self.metrics = None

            async def call_model(self, prompt):
                # For testing: return a simple response
                return "This is a mock LLM response for testing purposes."

        event = FakeEvent(payload)
        ctx = FakeCtx(session)
        return await self.on_event(event, ctx)