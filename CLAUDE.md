# MemoryMCP - Claude Documentation

## Overview

MemoryMCP is a Model Context Protocol (MCP) server that provides persistent memory and learning capabilities for AI assistants. It enables Claude and other AI agents to remember past interactions, learn from experiences, and maintain context across conversations.

## Current Status

✅ **Fully Implemented and Functional**

The MemoryMCP server is complete with all core features working:
- Semantic memory storage using local embeddings
- Multi-agent support with isolated memory stores  
- Context-aware memory capture
- Reflection and insight generation
- Memory export/import capabilities
- Advanced search by context and content

## Architecture

### Technology Stack
- **Embedding Model**: Sentence Transformers (all-MiniLM-L6-v2)
  - 384-dimensional embeddings
  - ~80MB model download on first run
  - Completely local - no API keys required
- **Storage**: SQLite with binary BLOB storage for embeddings
- **Framework**: MCP with FastMCP for tool definitions
- **Language**: Python 3.11+

### Module Structure

The codebase is organized into focused modules (~200 lines each):

```
memorymcp/
├── __init__.py       # Entry point (19 lines)
├── server.py         # MCP server and tool decorators (254 lines)
├── tools.py          # Tool implementation logic (284 lines)
├── memory_store.py   # SQLite storage backend (196 lines)
└── context_utils.py  # Context capture utilities (172 lines)
```

### Key Design Decisions

1. **Local Embeddings**: Switched from OpenAI to Sentence Transformers for privacy and cost
2. **Modular Architecture**: Split monolithic server.py into focused modules
3. **Async Support**: All memory operations are async-compatible
4. **Context Capture**: Automatic detection of languages, frameworks, and project context
5. **Memory Strength**: Implements memory strengthening on access (up to 10% boost per access)

## Usage with Claude

### Installation

```bash
# Clone and install
git clone https://github.com/yourusername/MemoryMCP.git
cd MemoryMCP
uv pip install -e .
```

### Configuration

Add to your Claude Desktop configuration:

```json
{
  "mcpServers": {
    "memory": {
      "command": "uv",
      "args": ["run", "memorymcp"],
      "cwd": "/path/to/MemoryMCP"
    }
  }
}
```

### Available Tools

1. **initialize_agent** - Set up agent memory store
2. **observe** - Store observations and experiences
3. **recall** - Retrieve relevant memories
4. **reflect** - Generate insights from memories
5. **list_agents** - List all agents
6. **export_memories** - Export memories (JSON/Markdown)
7. **search_by_context** - Advanced context search
8. **get_context_summary** - Get knowledge summary

## Implementation Details

### Memory Storage Schema

```sql
CREATE TABLE memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,        -- 384-dimensional vector
    strength REAL DEFAULT 1.0,      -- Memory importance/strength
    access_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
    context JSON,                   -- System and inferred context
    metadata JSON                   -- Tags, importance, type
)
```

### Scoring Algorithm

```python
semantic_similarity = cosine_similarity(query_embedding, memory_embedding)
days_old = (current_time - created_at).days
recency_score = exp(-days_old / 30)  # 50% decay after 30 days
final_score = (semantic_similarity * (1 - recency_weight) + 
               recency_score * recency_weight) * strength
```

### Context Capture

Automatically captures:
- System context (OS, Python version, working directory)
- Project indicators (.git, package.json, etc.)
- Detected languages and frameworks from content
- File references mentioned in observations

## Testing

```bash
# Run linting
uv run ruff check

# Format code  
uv run ruff format

# Test functionality (see test examples in previous conversations)
```

## Maintenance Notes

- The embedding model is cached after first download
- Memory stores are located at `~/.memorymcp/agents/`
- Each agent has its own SQLite database
- Memories strengthen by 10% on each access (capped at 1.0)
- No automatic pruning implemented yet (future enhancement)

## Future Enhancements

- [ ] Implement memory pruning (delete memories with strength < 0.1)
- [ ] Add time-based decay mechanism
- [ ] Support for episodic memory chains
- [ ] Memory compression for old memories
- [ ] Vector database integration for larger scale
- [ ] Memory visualization tools