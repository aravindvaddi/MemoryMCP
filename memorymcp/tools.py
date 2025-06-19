"""MCP tool definitions for memory operations."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from .memory_store import MemoryStore

# Base directory for all agent memory stores
MEMORY_BASE_DIR = Path.home() / ".memorymcp" / "agents"

# Global registry of active memory stores
memory_stores: dict[str, MemoryStore] = {}


def get_or_create_memory_store(agent_id: str) -> MemoryStore:
    """Get existing memory store or create a new one for an agent"""
    if agent_id not in memory_stores:
        memory_stores[agent_id] = MemoryStore(agent_id, MEMORY_BASE_DIR)
    return memory_stores[agent_id]


def initialize_agent_tool(
    agent_id: str = Field(description="Unique identifier for the agent instance"),
    personality: str | None = Field(
        None, description="Optional personality description or system prompt for the agent"
    ),
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata about the agent (e.g., capabilities, preferences)"
    ),
) -> str:
    """
    Initialize or connect to an agent's memory store.
    Creates a new agent profile if it doesn't exist.
    """
    agent_dir = MEMORY_BASE_DIR / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    # Store agent configuration
    config_path = agent_dir / "config.json"
    config = {
        "agent_id": agent_id,
        "created_at": datetime.utcnow().isoformat(),
        "personality": personality,
        "metadata": metadata or {},
    }

    if config_path.exists():
        # Load existing config
        with open(config_path) as f:
            existing_config = json.load(f)
            config["created_at"] = existing_config.get("created_at", config["created_at"])
            config["first_connected"] = existing_config.get("first_connected", config["created_at"])
        config["last_connected"] = datetime.utcnow().isoformat()
    else:
        config["first_connected"] = config["created_at"]
        config["last_connected"] = config["created_at"]

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Initialize memory store
    get_or_create_memory_store(agent_id)

    return f"Agent '{agent_id}' initialized. Memory store ready at {agent_dir}"


def list_agents_tool() -> list[dict[str, Any]]:
    """
    List all available agent instances with their metadata.
    Useful for discovering existing agents to connect to.
    """
    agents = []

    if not MEMORY_BASE_DIR.exists():
        return agents

    for agent_dir in MEMORY_BASE_DIR.iterdir():
        if agent_dir.is_dir():
            config_path = agent_dir / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    agents.append(
                        {
                            "agent_id": config["agent_id"],
                            "created_at": config.get("created_at"),
                            "last_connected": config.get("last_connected"),
                            "personality": config.get("personality"),
                            "metadata": config.get("metadata", {}),
                        }
                    )

    return sorted(agents, key=lambda x: x.get("last_connected", ""), reverse=True)


def export_memories_tool(
    agent_id: str = Field(description="The agent whose memories to export"),
    format: str = Field("json", description="Export format: 'json' or 'markdown'", pattern="^(json|markdown)$"),
    include_metadata: bool = Field(True, description="Whether to include metadata in the export"),
) -> str:
    """
    Export an agent's memories for backup or analysis.
    Returns the file path where memories were exported.
    """
    store = get_or_create_memory_store(agent_id)
    memories = store.get_recent_memories(limit=10000)  # Get all memories

    export_dir = MEMORY_BASE_DIR / agent_id / "exports"
    export_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    if format == "json":
        export_path = export_dir / f"memories_{timestamp}.json"
        export_data = {
            "agent_id": agent_id,
            "export_date": datetime.utcnow().isoformat(),
            "memory_count": len(memories),
            "memories": memories
            if include_metadata
            else [{"content": m["content"], "timestamp": m["timestamp"]} for m in memories],
        }
        with open(export_path, "w") as f:
            json.dump(export_data, f, indent=2)
    else:  # markdown
        export_path = export_dir / f"memories_{timestamp}.md"
        with open(export_path, "w") as f:
            f.write(f"# Memory Export for Agent: {agent_id}\n\n")
            f.write(f"Export Date: {datetime.utcnow().isoformat()}\n")
            f.write(f"Total Memories: {len(memories)}\n\n")

            for memory in memories:
                f.write(f"## {memory['timestamp']}\n\n")
                f.write(f"{memory['content']}\n\n")
                if include_metadata:
                    metadata = json.loads(memory.get("metadata", "{}"))
                    if metadata:
                        f.write(f"*Metadata: {json.dumps(metadata, indent=2)}*\n\n")
                f.write("---\n\n")

    return str(export_path)


def search_by_context_tool(
    agent_id: str = Field(description="The agent instance searching memories"),
    context_filters: dict[str, Any] | None = Field(
        None, description="Context filters (e.g., {'system.cwd': '/path', 'inferred.detected_languages': ['python']})"
    ),
    text_query: str | None = Field(None, description="Optional text search within matching memories"),
    limit: int = Field(10, description="Maximum number of results", ge=1, le=100),
) -> list[dict[str, Any]]:
    """
    Search memories by their captured context.
    Allows filtering by system context, inferred context, or any metadata.
    """
    store = get_or_create_memory_store(agent_id)

    # Get all memories (we'll filter in Python for now)
    # In production, you'd want to index the JSON metadata for efficient queries
    all_memories = store.get_recent_memories(limit=1000)

    filtered_memories = []

    for memory in all_memories:
        metadata = json.loads(memory.get("metadata", "{}"))
        context = metadata.get("context", {})

        # Check context filters
        if context_filters:
            match = True
            for filter_key, filter_value in context_filters.items():
                # Navigate nested context (e.g., "system.cwd")
                keys = filter_key.split(".")
                current = context

                for key in keys[:-1]:
                    if key in current:
                        current = current[key]
                    else:
                        match = False
                        break

                if match and keys[-1] in current:
                    actual_value = current[keys[-1]]

                    # Handle list contains check
                    if isinstance(filter_value, list) and isinstance(actual_value, list):
                        if not any(v in actual_value for v in filter_value):
                            match = False
                    # Handle exact match
                    elif actual_value != filter_value:
                        match = False
                else:
                    match = False

                if not match:
                    break

            if not match:
                continue

        # Check text query
        if text_query and text_query.lower() not in memory["content"].lower():
            continue

        # Add to results
        filtered_memories.append(
            {
                "id": memory["id"],
                "timestamp": memory["timestamp"],
                "content": memory["content"],
                "metadata": metadata,
                "context_summary": {
                    "cwd": context.get("system", {}).get("cwd", "unknown"),
                    "languages": context.get("inferred", {}).get("detected_languages", []),
                    "frameworks": context.get("inferred", {}).get("detected_frameworks", []),
                    "files": len(context.get("related_files", [])),
                },
            }
        )

    # Sort by timestamp (most recent first)
    filtered_memories.sort(key=lambda x: x["timestamp"], reverse=True)

    return filtered_memories[:limit]


def get_context_summary_tool(agent_id: str = Field(description="The agent instance to analyze")) -> dict[str, Any]:
    """
    Get a summary of all contexts captured for an agent's memories.
    Useful for understanding the agent's knowledge domains and work history.
    """
    store = get_or_create_memory_store(agent_id)
    memories = store.get_recent_memories(limit=1000)

    summary = {
        "total_memories": len(memories),
        "contexts": {"working_directories": {}, "languages": {}, "frameworks": {}, "concepts": {}, "file_types": {}},
        "time_range": {"earliest": None, "latest": None},
    }

    for memory in memories:
        metadata = json.loads(memory.get("metadata", "{}"))
        context = metadata.get("context", {})

        # Update time range
        timestamp = memory["timestamp"]
        if not summary["time_range"]["earliest"] or timestamp < summary["time_range"]["earliest"]:
            summary["time_range"]["earliest"] = timestamp
        if not summary["time_range"]["latest"] or timestamp > summary["time_range"]["latest"]:
            summary["time_range"]["latest"] = timestamp

        # Count working directories
        cwd = context.get("system", {}).get("cwd")
        if cwd:
            summary["contexts"]["working_directories"][cwd] = summary["contexts"]["working_directories"].get(cwd, 0) + 1

        # Count languages, frameworks, concepts
        inferred = context.get("inferred", {})
        for lang in inferred.get("detected_languages", []):
            summary["contexts"]["languages"][lang] = summary["contexts"]["languages"].get(lang, 0) + 1

        for framework in inferred.get("detected_frameworks", []):
            summary["contexts"]["frameworks"][framework] = summary["contexts"]["frameworks"].get(framework, 0) + 1

        for concept in inferred.get("detected_concepts", []):
            summary["contexts"]["concepts"][concept] = summary["contexts"]["concepts"].get(concept, 0) + 1

        # Count file types
        for file_ref in context.get("related_files", []):
            if "." in file_ref:
                ext = file_ref.split(".")[-1].lower()
                summary["contexts"]["file_types"][ext] = summary["contexts"]["file_types"].get(ext, 0) + 1

    # Sort contexts by frequency
    for key in summary["contexts"]:
        summary["contexts"][key] = dict(sorted(summary["contexts"][key].items(), key=lambda x: x[1], reverse=True))

    return summary

