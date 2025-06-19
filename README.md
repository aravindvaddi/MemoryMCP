# MemoryMCP

A Model Context Protocol (MCP) server that provides persistent memory and learning capabilities for AI assistants. MemoryMCP enables AI agents to remember past interactions, learn from experiences, and maintain context across conversations.

## Features

- **Semantic Memory Storage**: Store observations with local embeddings for intelligent retrieval (no API keys required!)
- **Multi-Agent Support**: Each agent maintains its own isolated memory store
- **Smart Retrieval**: Find relevant memories using semantic search with recency weighting
- **Memory Strength**: Memories strengthen with access and importance ratings
- **Context Awareness**: Automatic capture of system context, project details, and conversation patterns
- **Reflection Capabilities**: Agents can analyze their memories to form higher-level insights
- **Privacy-First**: All processing happens locally using Sentence Transformers
- **Export/Import**: Backup and restore memory stores in JSON or Markdown format
- **Advanced Search**: Search by context, tags, or semantic similarity

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MemoryMCP.git
cd MemoryMCP
```

2. Install dependencies using UV:
```bash
uv pip install -e .
```

Note: The first run will download the embedding model (~80MB), which will be cached locally.

## Usage

### Running the Server

The MCP server can be integrated with any MCP-compatible client.

```bash
uv run memorymcp
```

### Core Tools

1. **initialize_agent** - Create or connect to an agent's memory store
   ```json
   {
     "agent_id": "assistant_1",
     "personality": "A helpful coding assistant",
     "metadata": {"capabilities": ["python", "web"]}
   }
   ```

2. **observe** - Store new memories with automatic embedding generation
   ```json
   {
     "agent_id": "assistant_1",
     "content": "User prefers TypeScript over JavaScript for new projects",
     "importance": 0.8,
     "tags": ["preferences", "typescript"],
     "context": {"project": "web-app"}
   }
   ```

3. **recall** - Retrieve relevant memories using semantic search
   ```json
   {
     "agent_id": "assistant_1", 
     "query": "What does the user think about TypeScript?",
     "limit": 10,
     "recency_bias": true
   }
   ```

4. **reflect** - Generate insights from past experiences
   ```json
   {
     "agent_id": "assistant_1",
     "topic": "user coding preferences",
     "depth": 20
   }
   ```

5. **list_agents** - List all available agent instances
   ```json
   {}
   ```

6. **export_memories** - Export memories for backup or analysis
   ```json
   {
     "agent_id": "assistant_1",
     "format": "json",
     "include_metadata": true
   }
   ```

7. **search_by_context** - Search memories by their captured context
   ```json
   {
     "agent_id": "assistant_1",
     "context_filters": {"inferred.detected_languages": ["python"]},
     "text_query": "debugging",
     "limit": 10
   }
   ```

8. **get_context_summary** - Get a summary of agent's knowledge domains
   ```json
   {
     "agent_id": "assistant_1"
   }
   ```

## Architecture

MemoryMCP uses:
- **SQLite** for efficient local storage with full-text search
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic embeddings - no API keys needed!
- **Binary BLOB storage** for embedding vectors (384 dimensions)
- **JSON metadata** for flexible context storage
- **Modular design** with clean separation of concerns

### File Structure
```
memorymcp/
├── __init__.py       # Entry point
├── server.py         # MCP server and tool decorators (254 lines)
├── tools.py          # Tool implementation logic (284 lines)
├── memory_store.py   # SQLite storage backend (196 lines)
└── context_utils.py  # Context capture utilities (172 lines)
```

### Memory Scoring

Memories are scored based on:
- **Semantic Similarity**: How closely the memory matches the query
- **Recency**: More recent memories get higher scores (exponential decay)
- **Strength**: Important memories and frequently accessed memories are stronger

Score = (Similarity × (1 - RecencyWeight) + RecencyScore × RecencyWeight) × Strength

## Memory Storage

Memories are stored in SQLite with the following schema:
- Unique ID and agent association
- Content and embedding vector
- Strength and access count
- Creation and last accessed timestamps
- JSON context and metadata

Context automatically captured includes:
- System information (OS, Python version, etc.)
- Working directory and project indicators
- Detected programming languages and frameworks
- File references mentioned in content

## Development

### Running Tests
```bash
uv run pytest
```

### Code Quality
```bash
uv run ruff check
uv run ruff format
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.