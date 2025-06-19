# MemoryMCP

A Model Context Protocol (MCP) server that provides persistent memory and learning capabilities for AI assistants. MemoryMCP enables AI agents to remember past interactions, learn from experiences, and maintain context across conversations.

## Features

- **Semantic Memory Storage**: Store observations with local embeddings for intelligent retrieval (no API keys required!)
- **Multi-Agent Support**: Each agent maintains its own isolated memory store
- **Smart Retrieval**: Find relevant memories using semantic search with recency weighting
- **Memory Decay**: Natural forgetting with strength-based memory persistence
- **Context Awareness**: Automatic capture of system context, project details, and conversation patterns
- **Reflection Capabilities**: Agents can analyze their memories to form higher-level insights
- **Privacy-First**: All processing happens locally - no external API calls

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

### Core Tools

1. **initialize_agent** - Create or connect to an agent's memory store
   ```json
   {
     "agent_id": "assistant_1",
     "personality": "A helpful coding assistant"
   }
   ```

2. **observe** - Store new memories with automatic embedding generation
   ```json
   {
     "agent_id": "assistant_1",
     "content": "User prefers TypeScript over JavaScript for new projects",
     "importance": 0.8,
     "tags": ["preferences", "typescript"]
   }
   ```

3. **recall** - Retrieve relevant memories using semantic search
   ```json
   {
     "agent_id": "assistant_1", 
     "query": "What does the user think about TypeScript?",
     "limit": 10
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

## Architecture

MemoryMCP uses:
- **SQLite** for efficient local storage
- **Sentence Transformers** (all-MiniLM-L6-v2) for semantic search - no API keys needed!
- **Binary BLOB storage** for embedding vectors (384 dimensions)
- **Automatic context capture** for rich memory metadata
- **100% local processing** - your data never leaves your machine

See `MEMORY_DESIGN.md` for detailed architecture documentation.

## License

GNU Affero General Public License v3.0