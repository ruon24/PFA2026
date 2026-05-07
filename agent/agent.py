"""LiveKit worker: a Tunisian Derja voice tutor backed by the PFA2026 RAG.

Run from the PFA2026/ directory (where chroma_db/ lives):

    uv run python -m agent.agent dev      # local dev mode (joins LiveKit room)
    uv run python -m agent.agent start    # production mode

Requires .env.local with GOOGLE_API_KEY and LIVEKIT_URL/API_KEY/API_SECRET.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    function_tool,
)
from livekit.plugins import google

from agent.kb import KnowledgeBase
from agent.prompts import DERJA_TUTOR_INSTRUCTIONS

# Load env from the agent/ directory (next to this file).
# Tries .env.local first, falls back to .env. Both are gitignored.
_AGENT_DIR = Path(__file__).resolve().parent
load_dotenv(_AGENT_DIR / ".env.local")
load_dotenv(_AGENT_DIR / ".env")

logger = logging.getLogger("derja-tutor")
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))

KB = KnowledgeBase()
logger.info("KnowledgeBase ready: %s", KB.stats())


class DerjaTutor(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=DERJA_TUTOR_INSTRUCTIONS)

    @function_tool()
    async def search_knowledge_base(
        self,
        context: RunContext,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Search the Tunisian school manuals (grades 4-6) for relevant passages.

        Call this every time the student asks a content question (geography,
        science, history, math, etc). Translate the student's Derja question
        into French keywords first — the manuals are written in French.

        Args:
            query: Short search phrase, ideally in French.
            top_k: How many passages to return. Default 3, max useful ~5.
        """
        results = KB.search(query, top_k=top_k)
        logger.info("search_knowledge_base(query=%r, top_k=%d) -> %d chunks",
                    query, top_k, len(results))
        return results

    @function_tool()
    async def list_available_subjects(self, context: RunContext) -> dict[str, list[str]]:
        """List the source filenames in the knowledge base, grouped by grade.

        Use this when the student asks what topics or grades you can help with.
        """
        subjects = KB.list_subjects()
        logger.info("list_available_subjects -> grades=%s", list(subjects.keys()))
        return subjects


server = AgentServer()


@server.rtc_session(agent_name="derja-tutor")
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession(
        llm=google.realtime.RealtimeModel(
            model="gemini-2.5-flash-native-audio-preview-12-2025",
            voice="Aoede",
            temperature=0.7,
        ),
    )
    await session.start(agent=DerjaTutor(), room=ctx.room)
    await session.generate_reply(
        instructions=(
            "Sallem 3al user bel-derja, 9oul-lou enti m3awna lel etudiants "
            "ta3 4ème, 5ème w 6ème ابتدائي, w as2lou 9ifech tnejjem t3awnou "
            "el-yowm. Khallila el-jawebek 9sira (jumlatin)."
        )
    )


if __name__ == "__main__":
    agents.cli.run_app(server)
