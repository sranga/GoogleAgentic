"""
Clinic Finder Agent.

Responsibilities:
- Use built-in GoogleSearch / Maps tools when available to find nearby clinics.
- Prefetch availability in parallel for top candidates (demonstrates parallel agents).
- Return a list of candidate clinics to session for downstream scheduling.
- Provide robust fallback (mock data) if tools are not configured.
- Emit logs and simple metrics through a MetricsCollector if present.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

# Try to import ADK tool classes; fallback to placeholders if not present
try:
    from adk import Agent, EventActions
    from adk.tools import GoogleSearchTool, GoogleMapsTool
except Exception:
    # Minimal placeholder Agent so this file can still be used in unit tests.
    from adk import Agent, EventActions  # type: ignore
    GoogleSearchTool = None  # type: ignore
    GoogleMapsTool = None  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ClinicFinderAgent(Agent):
    def __init__(self, config: Dict[str, Any]):
        """
        config keys:
          - model: model name for LLM calls (if used)
          - max_candidates: how many candidate clinics to return
          - tools_enabled: True/False to use Google tools if available
        """
        super().__init__(
            name="clinic_finder_agent",
            model=config.get("model"),
            description=(
                "Finds nearby clinics and prefetches availability. "
                "Prefer Google Maps/Search tools when configured; otherwise returns mock data."
            ),
            instructions=(
                "Return a list of candidate clinics with id, name, distance_km, has_api flag."
            ),
        )
        self.config = config
        self.max_candidates = config.get("max_candidates", 3)
        self.tools_enabled = config.get("tools_enabled", True)

        # Attempt to register tools if available
        self.google_search = None
        self.google_maps = None
        if self.tools_enabled:
            try:
                if GoogleSearchTool:
                    self.google_search = GoogleSearchTool(name="google_search")
                if GoogleMapsTool:
                    self.google_maps = GoogleMapsTool(name="google_maps")
            except Exception:
                logger.warning("Google Search/Maps tools not available; proceeding with fallback")

    async def on_event(self, event, ctx):
        session = ctx.session
        query = session.get("location_query") or event.payload.get("location_query") if event.payload else None
        logger.info("ClinicFinderAgent invoked; query=%s", query)

        # Primary path: use Maps API if available
        if self.google_maps:
            try:
                candidates = await self._find_with_maps(ctx, query)
            except Exception as e:
                logger.exception("Maps tool failed: %s", e)
                candidates = self._mock_candidates()
        elif self.google_search:
            try:
                candidates = await self._find_with_search(ctx, query)
            except Exception as e:
                logger.exception("Search tool failed: %s", e)
                candidates = self._mock_candidates()
        else:
            candidates = self._mock_candidates()

        # Save to session and optionally prefetch availability in parallel
        session["last_clinics"] = candidates
        # Start background prefetch of availability for top candidates (non-blocking)
        asyncio.create_task(self._prefetch_availability(ctx, candidates[: self.max_candidates]))

        # Observability: log metric via ctx.metrics if available
        try:
            if hasattr(ctx, "metrics") and ctx.metrics:
                ctx.metrics.increment("clinic_searches")
        except Exception:
            pass

        return EventActions(resume=True, message={"candidates": candidates})

    async def _find_with_maps(self, ctx, query: Optional[str]) -> List[Dict[str, Any]]:
        """Use Google Maps tool via ctx.call_tool or tool wrapper (ADK dependent)."""
        logger.info("Finding clinics via Google Maps: query=%s", query)
        # Build params for maps search; this will depend on your Maps tool signature
        params = {"query": query or "vaccination clinic near me", "radius_km": 10}
        try:
            # ctx.call_tool is ADK helper to invoke tools registered on agent
            result = await ctx.call_tool("google_maps", params)
            # Normalize result into candidate list
            candidates = []
            for place in result.get("places", [])[: self.max_candidates]:
                candidates.append(
                    {
                        "id": place.get("place_id"),
                        "name": place.get("name"),
                        "
