# Memory MCP Design Document

## Overview

Memory MCP is a Model Context Protocol server that provides persistent memory capabilities for AI agents. It enables agents to store observations from interactions and recall relevant memories based on semantic similarity, recency, and context.

## Implementation Note

This document describes the original design. The actual implementation differs in some key areas:
- **Embeddings**: Uses Sentence Transformers (384 dimensions) instead of OpenAI (1536 dimensions)
- **Tool Parameters**: Some parameter names differ (e.g., `content` vs `observation`)
- **Memory Decay**: Currently only implements strength boost on access, not time-based decay
- **Additional Features**: The implementation includes extra tools like `list_agents`, `export_memories`, `search_by_context`, and `get_context_summary`

See README.md and CLAUDE.md for current implementation details.

## Core Concept

The system mimics human memory patterns:
- Recent memories are vivid and detailed
- Older memories fade but can be triggered by strong associations
- Context matters - memories from similar situations are more relevant
- Important or frequently accessed memories persist longer

## Architecture

### Storage Design

**SQLite Database Schema:**
```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,  -- 1536-dim vector stored as binary (Note: Implementation uses 384-dim)
    strength REAL DEFAULT 1.0,
    access_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    context JSON,
    metadata JSON
);

CREATE INDEX idx_agent_id ON memories(agent_id);
CREATE INDEX idx_created_at ON memories(created_at);
CREATE INDEX idx_strength ON memories(strength);
```

**File Structure:**
```
~/.memorymcp/agents/
├── [agent_id]/
│   ├── config.json       # Agent configuration
│   ├── memories.db       # SQLite database
│   └── exports/          # Backup directory
```

### Memory Lifecycle

1. **Creation**: When `observe()` is called, the system:
   - Generates embedding using OpenAI text-embedding-3-small (1536 dimensions)
   - Captures automatic context (cwd, timestamp, environment)
   - Stores with initial strength value

2. **Retrieval**: When `recall()` is called, the system:
   - Embeds the search query
   - Calculates relevance score for each memory:
     ```
     score = semantic_similarity * 0.5 +
             recency_score * 0.3 +
             context_overlap * 0.2
     ```
   - Returns top memories above threshold

3. **Decay**: Memory strength decays over time:
   - `strength = initial_strength * (0.95 ^ days_elapsed)`
   - Memories below 0.1 strength are candidates for pruning
   - Access boosts strength: `strength *= 1.1` per access

### MCP Tool Interface

```python
@mcp.tool()
async def initialize_agent(
    agent_id: str,
    description: str = ""
) -> str:
    """Create or connect to an agent's memory store"""

@mcp.tool()
async def observe(
    agent_id: str,
    observation: str,
    importance: float = 1.0,
    context: Optional[Dict] = None
) -> str:
    """Store an observation in agent's memory"""

@mcp.tool()
async def recall(
    agent_id: str,
    trigger: str,
    limit: int = 10
) -> List[Dict]:
    """Retrieve relevant memories based on semantic search"""

@mcp.tool()
async def reflect(
    agent_id: str,
    topic: Optional[str] = None
) -> str:
    """Generate insights from memory patterns"""
```

### Context Capture

Automatic context includes:
- System: cwd, platform, timestamp, environment
- Inferred: detected language, frameworks, concepts from content
- Explicit: LLM-provided task_id, pr_number, etc.
- Files: recently modified files in working directory

### Performance Characteristics

**Storage Efficiency:**
- Per memory: ~7-8 KB (6KB embedding + metadata)
- 10K memories: ~70-80 MB
- 50K memories: ~350-400 MB

**Query Performance:**
- 1K memories: 10-20ms
- 10K memories: 100-200ms
- 50K memories: 500-1000ms

**Scaling Strategy:**
- SQLite handles up to 50K memories efficiently
- Beyond 50K: implement rolling window or archive old memories
- Future: migrate to vector database (Chroma/Pinecone) if needed

### Implementation Notes

1. **Embedding Storage**: Use binary BLOB format with struct.pack
2. **Similarity Calculation**: NumPy cosine similarity in-memory
3. **Batch Processing**: Load memories in 1000-record chunks
4. **Connection Pooling**: One SQLite connection per agent
5. **Thread Safety**: Use connection-per-request pattern

### Memory Pruning

When database exceeds 50K memories:
1. Delete memories with strength < 0.1
2. Archive memories older than 6 months
3. Keep all memories marked as "important" (importance > 0.8)

### Security Considerations

- Agent IDs should be validated (alphanumeric + underscore only)
- Memory content is not encrypted (assume local trust)
- No cross-agent memory access without explicit permission

## Usage Example

```python
# Initialize agent
await initialize_agent("project_assistant", "Helps with Python projects")

# Store observation
await observe(
    agent_id="project_assistant",
    observation="User prefers pytest over unittest for this project",
    importance=0.8,
    context={"project": "auth_service", "file": "tests/test_auth.py"}
)

# Recall relevant memories
memories = await recall(
    agent_id="project_assistant",
    trigger="How should I write tests for this service?"
)
# Returns: ["User prefers pytest over unittest...", ...]
```

## Future Enhancements

1. **Memory Consolidation**: Periodically merge similar memories
2. **Cross-Agent Learning**: Share insights between agents
3. **Episodic Memories**: Group related observations into episodes
4. **Causal Chains**: Track cause-effect relationships in memories