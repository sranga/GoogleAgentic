# app.py

from vertexai.agent_engines import AdkApp
from vaccess_agent import VAccessOrchestrator
from config import CONFIG

orchestrator_instance = VAccessOrchestrator(CONFIG)
app = AdkApp(agent=orchestrator_instance)

# Test locally
if __name__ == "__main__":
    import asyncio


    async def test():
        async for event in app.async_stream_query(
                user_id="test_user",
                message="Find me a clinic in 94024"
        ):
            print(event)


    asyncio.run(test())