"""Backward-compatible import surface for the relocated Google client."""

from trae_agent.utils.llm_clients.google_client import GoogleClient, genai

__all__ = ["GoogleClient", "genai"]
