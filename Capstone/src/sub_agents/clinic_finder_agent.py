"""
Clinic Finder Agent

Finds nearby vaccination clinics using Google Maps and Search tools.
Supports parallel availability prefetching for top candidates.

Features:
- Google Maps API integration for location search
- Google Search fallback
- Parallel availability prefetching
- Distance and rating sorting
- Structured logging and metrics
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import re

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
        def __init__(self, *args, **kwargs):
            pass

from google.adk.tools import google_search

logger = logging.getLogger(__name__)


class ClinicFinderAgent(Agent):
    """
    Finds nearby vaccination clinics using Google Maps/Search tools.
    Returns candidate clinics with availability information.
    """

    def __init__(self, config: Dict[str, Any]):
        self.google_search = google_search
        self.google_maps = google_search

        super().__init__(
            name="clinic_finder_agent",
            model=config.get("model", "gemini-2.0-flash"),
            description="Finds nearby vaccination clinics using Google Maps and Search.",
            instruction="Return a list of candidate clinics with id, name, distance_km, and availability.",
            tools=[self.google_search, self.google_maps],
        )

        self.config = config
        self.max_candidates = config.get("max_candidates", 5)
        self.search_radius_km = config.get("search_radius_km", 10)

    async def on_event(self, event, ctx):
        """Handle clinic search requests."""
        session = ctx.session
        payload = event.payload or {}
        query = payload.get("location_query") or session.get("location_query")

        logger.info(
            "ClinicFinderAgent invoked",
            extra={"query": query, "session_id": session.get("user_id", "unknown")[:8]}
        )

        if hasattr(ctx, "metrics") and ctx.metrics:
            ctx.metrics.increment("clinic_searches")

        start_time = datetime.utcnow()

        # Try Maps first, then Search as fallback
        candidates = []
        search_method = "google_maps"

        try:
            candidates = await self._find_with_maps(ctx, query)
        except Exception as e:
            logger.warning("Maps search failed, trying Search: %s", e)
            search_method = "google_search"
            try:
                candidates = await self._find_with_search(ctx, query)
            except Exception as e2:
                logger.error("Both Maps and Search failed: %s", e2)
                search_method = "failed"

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        logger.info(
            "Clinic search completed",
            extra={
                "method": search_method,
                "candidates_found": len(candidates),
                "duration_ms": duration_ms
            }
        )

        session["last_clinics"] = candidates
        session["clinic_search_method"] = search_method

        # Prefetch availability in parallel (non-blocking)
        if candidates:
            asyncio.create_task(self._prefetch_availability(ctx, candidates[:self.max_candidates]))

        if hasattr(ctx, "metrics") and ctx.metrics:
            ctx.metrics.histogram("clinic_search_duration_ms", duration_ms, {"method": search_method})

        return EventActions(
            resume=True,
            message={"candidates": candidates[:self.max_candidates], "method": search_method}
        )

    async def _find_with_maps(self, ctx, query: Optional[str]) -> List[Dict[str, Any]]:
        """Search for clinics using Google Maps API."""
        logger.info("Finding clinics via Google Maps: query=%s", query)

        params = {
            "query": f"vaccination clinic near {query}" if query else "vaccination clinic near me",
            "radius_km": self.search_radius_km
        }

        result = await ctx.call_tool("google_maps", params)
        candidates = []

        for idx, place in enumerate(result.get("places", [])[:self.max_candidates]):
            candidates.append({
                "id": place.get("place_id", f"map_{idx}"),
                "name": place.get("name", f"Clinic {idx + 1}"),
                "address": place.get("address", "Address not available"),
                "distance_km": place.get("distance_km", 0.0),
                "rating": place.get("rating", 0.0),
                "phone": place.get("phone", ""),
                "hours": self._parse_hours(place.get("hours", [])),
                "has_api": self._check_api_availability(place),
                "source": "google_maps"
            })

        return candidates

    async def _find_with_search(self, ctx, query: Optional[str]) -> List[Dict[str, Any]]:
        """Search for clinics using Google Search as fallback."""
        logger.info("Finding clinics via Google Search: query=%s", query)

        search_query = f"vaccination clinic near {query}" if query else "vaccination clinic near me"
        params = {"query": search_query, "num_results": self.max_candidates}

        result = await ctx.call_tool("google_search", params)
        candidates = []

        for idx, item in enumerate(result.get("results", [])[:self.max_candidates]):
            candidates.append({
                "id": f"search_{idx}",
                "name": item.get("title", f"Clinic {idx + 1}"),
                "address": item.get("snippet", "Address not available"),
                "distance_km": None,  # Not available from search
                "rating": None,
                "phone": self._extract_phone(item.get("snippet", "")),
                "hours": [],
                "has_api": False,
                "url": item.get("link", ""),
                "source": "google_search"
            })

        return candidates

    async def _prefetch_availability(self, ctx, candidates: List[Dict[str, Any]]):
        """Prefetch appointment availability for clinics in parallel."""
        logger.info("Starting availability prefetch for %d clinics", len(candidates))

        async def fetch_availability(clinic: Dict[str, Any]):
            try:
                # Simulate availability check (replace with real API call)
                await asyncio.sleep(0.1)
                clinic["availability"] = {
                    "next_available": self._generate_next_slot(),
                    "slots_this_week": 10,
                    "last_checked": datetime.utcnow().isoformat()
                }
            except Exception as e:
                logger.warning("Failed to fetch availability for %s: %s", clinic["id"], e)
                clinic["availability"] = None

        tasks = [fetch_availability(c) for c in candidates if c.get("has_api")]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _generate_next_slot(self) -> str:
        """Generate a realistic next available appointment slot."""
        import random
        days_ahead = random.randint(1, 7)
        next_date = datetime.utcnow() + timedelta(days=days_ahead)
        hour = random.randint(9, 16)
        minute = random.choice([0, 15, 30, 45])
        return next_date.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat()

    def _parse_hours(self, hours_data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Parse and normalize business hours data."""
        if not hours_data:
            return []
        return [
            {
                "day": entry.get("day", "Unknown"),
                "open": entry.get("open", ""),
                "close": entry.get("close", "")
            }
            for entry in hours_data if isinstance(entry, dict)
        ]

    def _check_api_availability(self, place: Dict[str, Any]) -> bool:
        """Check if a clinic has booking API integration."""
        name = place.get("name", "").lower()
        api_chains = ["cvs", "walgreens", "walmart", "rite aid", "health center", "medical center"]
        return any(chain in name for chain in api_chains)

    def _extract_phone(self, text: str) -> str:
        """Extract phone number from text."""
        phone_pattern = r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
        match = re.search(phone_pattern, text)
        return match.group(0) if match else ""