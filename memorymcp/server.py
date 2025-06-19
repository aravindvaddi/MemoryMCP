"""Main MCP server implementation for MemoryMCP."""

import json
from typing import Any

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from sentence_transformers import SentenceTransformer

from .context_utils import (
    capture_system_context,
)
from .tools import (
    export_memories_tool,
    get_context_summary_tool,
    get_or_create_memory_store,
    initialize_agent_tool,
    list_agents_tool,
    search_by_context_tool,
)

# Load environment variables
load_dotenv()

# Create an MCP server
mcp = FastMCP("Memory")

# Initialize embedding model (using a small, efficient model)
embedding_model = None


def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Using all-MiniLM-L6-v2: small, fast, good quality
        # 384 dimensions, ~80MB download
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return embedding_model


@mcp.tool()
def initialize_agent(
    agent_id: str = Field(description="Unique identifier for the agent instance"),
    personality: str | None = Field(
        None, description="Optional personality description or system prompt for the agent"
    ),
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata about the agent (e.g., capabilities, preferences)"
    ),
) -> str:
    """Initialize or connect to an agent's memory store."""
    return initialize_agent_tool(agent_id, personality, metadata)


@mcp.tool()
async def observe(
    agent_id: str = Field(description="The agent instance making the observation"),
    content: str = Field(description="The observation or experience to remember"),
    importance: float | None = Field(None, description="Importance score (0-1) for the memory", ge=0, le=1),
    tags: list[str] | None = Field(None, description="Tags to categorize the memory"),
    context: dict[str, Any] | None = Field(None, description="Additional context about the observation"),
) -> str:
    """
    Store an observation or experience in the agent's memory.
    This is how agents build their long-term memory.
    """
    store = get_or_create_memory_store(agent_id)

    # Generate embedding for the content
    try:
        model = get_embedding_model()
        # Generate embedding (sentence-transformers handles batching internally)
        embedding = model.encode(content, convert_to_numpy=True)
    except Exception as e:
        return f"Error generating embedding: {str(e)}"

    # Capture automatic context
    auto_context = capture_system_context() if context is None else context

    metadata = {"importance": importance or 0.5, "tags": tags or [], "type": "observation"}

    memory_id = await store.add_memory(content, embedding, auto_context, metadata)

    return f"Memory #{memory_id} stored for agent '{agent_id}'"


@mcp.tool()
async def recall(
    agent_id: str = Field(description="The agent instance recalling memories"),
    query: str | None = Field(None, description="Search query to find relevant memories"),
    limit: int = Field(10, description="Maximum number of memories to recall", ge=1, le=50),
    recency_bias: bool = Field(True, description="Whether to prioritize recent memories"),
) -> list[dict[str, Any]]:
    """
    Recall memories from the agent's memory store.
    Returns relevant memories based on the query or recent memories if no query.
    """
    store = get_or_create_memory_store(agent_id)

    if query:
        # Generate embedding for the query
        try:
            model = get_embedding_model()
            query_embedding = model.encode(query, convert_to_numpy=True)
        except Exception:
            # Fallback to recent memories if embedding fails
            memories = store.get_recent_memories(limit)
        else:
            # Use semantic search
            recency_weight = 0.3 if recency_bias else 0.0
            memories = await store.search_memories(query_embedding, limit, recency_weight)
    else:
        memories = store.get_recent_memories(limit)

    # Format memories for return
    formatted_memories = []
    for memory in memories:
        # Handle both formats (semantic search returns full data, recent returns basic)
        if isinstance(memory, dict) and "metadata" in memory and isinstance(memory["metadata"], dict):
            # From semantic search
            formatted_memories.append(
                {
                    "id": memory["id"],
                    "timestamp": memory.get("created_at", memory.get("timestamp", "")),
                    "content": memory["content"],
                    "importance": memory["metadata"].get("importance", 0.5),
                    "tags": memory["metadata"].get("tags", []),
                    "context": memory.get("context", {}),
                    "score": memory.get("score", 0.0),
                }
            )
        else:
            # From get_recent_memories
            metadata = json.loads(memory.get("metadata", "{}")) if "metadata" in memory else {}
            formatted_memories.append(
                {
                    "id": memory["id"],
                    "timestamp": memory.get("created_at", memory.get("timestamp", "")),
                    "content": memory["content"],
                    "importance": metadata.get("importance", 0.5),
                    "tags": metadata.get("tags", []),
                    "context": json.loads(memory.get("context", "{}")) if "context" in memory else {},
                    "score": 0.0,
                }
            )

    return formatted_memories


@mcp.tool()
async def reflect(
    agent_id: str = Field(description="The agent instance performing reflection"),
    topic: str = Field(description="Topic or theme to reflect on"),
    depth: int = Field(20, description="Number of memories to consider for reflection", ge=5, le=100),
) -> str:
    """
    Generate insights by reflecting on past memories related to a topic.
    This helps agents form higher-level understanding from their experiences.
    """
    store = get_or_create_memory_store(agent_id)

    # Generate embedding for the topic
    try:
        model = get_embedding_model()
        topic_embedding = model.encode(topic, convert_to_numpy=True)
    except Exception as e:
        return f"Error generating embedding for reflection: {str(e)}"

    # Search for memories related to the topic
    memories = await store.search_memories(topic_embedding, depth)

    if not memories:
        return f"No memories found related to '{topic}' for agent '{agent_id}'"

    # Create a reflection summary
    reflection_parts = [f"Reflection on '{topic}' based on {len(memories)} memories:", ""]

    # Group memories by tags or patterns
    tag_groups = {}
    for memory in memories:
        tags = memory.get("metadata", {}).get("tags", ["untagged"])
        for tag in tags:
            if tag not in tag_groups:
                tag_groups[tag] = []
            tag_groups[tag].append(memory["content"])

    # Summarize patterns
    if tag_groups:
        reflection_parts.append("Patterns observed:")
        for tag, contents in tag_groups.items():
            reflection_parts.append(f"- {tag}: {len(contents)} related memories")

    # Generate embedding for the reflection content
    reflection_content = (
        f"Reflected on '{topic}': Found {len(memories)} relevant memories with patterns in {list(tag_groups.keys())}"
    )

    try:
        reflection_embedding = model.encode(reflection_content, convert_to_numpy=True)
    except Exception as e:
        return f"Error storing reflection: {str(e)}"

    # Store the reflection itself as a memory
    reflection_metadata = {
        "type": "reflection",
        "topic": topic,
        "memory_count": len(memories),
        "tags": ["reflection", topic],
        "importance": 0.8,  # Reflections are important
    }

    await store.add_memory(reflection_content, reflection_embedding, capture_system_context(), reflection_metadata)

    reflection_parts.append("")
    reflection_parts.append("Reflection stored as new memory for future reference.")

    return "\n".join(reflection_parts)


@mcp.tool()
def list_agents() -> list[dict[str, Any]]:
    """List all available agent instances with their metadata."""
    return list_agents_tool()


@mcp.tool()
def export_memories(
    agent_id: str = Field(description="The agent whose memories to export"),
    format: str = Field("json", description="Export format: 'json' or 'markdown'", pattern="^(json|markdown)$"),
    include_metadata: bool = Field(True, description="Whether to include metadata in the export"),
) -> str:
    """Export an agent's memories for backup or analysis."""
    return export_memories_tool(agent_id, format, include_metadata)


@mcp.tool()
def search_by_context(
    agent_id: str = Field(description="The agent instance searching memories"),
    context_filters: dict[str, Any] | None = Field(
        None, description="Context filters (e.g., {'system.cwd': '/path', 'inferred.detected_languages': ['python']})"
    ),
    text_query: str | None = Field(None, description="Optional text search within matching memories"),
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100),
) -> list[dict[str, Any]]:
    """Search memories by their captured context."""
    return search_by_context_tool(agent_id, context_filters, text_query, limit)


@mcp.tool()
def get_context_summary(agent_id: str = Field(description="The agent instance to analyze")) -> dict[str, Any]:
    """Get a summary of all contexts captured for an agent's memories."""
    return get_context_summary_tool(agent_id)

