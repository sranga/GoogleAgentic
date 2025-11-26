# app.py

from google.adk.agent_engines import AdkApp, AgentType
from vaccess_agent import VAccessOrchestrator
from config import CONFIG
import sys

orchestrator_instance = VAccessOrchestrator(CONFIG)
app = AdkApp(
    agent=orchestrator_instance,
    supported_agent_classes={
        VAccessOrchestrator: AgentType.AGENT
    }
)