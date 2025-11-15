"""
VaccineInfoAgent

Responsibilities:
- Provide clear, factual vaccine information.
- Handle user education questions (eligibility, side-effects, safety).
- Use MemoryBank signals (preferred language, prior questions).
- Produce stable, safe answers (no medical diagnosis).
- Compact session context before LLM calls.
- Demonstrates context engineering + observability.

This agent does NOT hallucinate — it uses a stable knowledge base + LLM expansion.
"""

import logging
from typing import Dict, Any, List

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
        "and
