# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MemoryMCP is an MCP (Model Context Protocol) server implementation focused on memory-related functionality. The project uses FastMCP to create an MCP server that can interact with language models.

## Technology Stack

- **Language**: Python 3.11
- **Framework**: FastMCP (MCP server framework)
- **Package Manager**: UV (modern Python package manager)

## Development Commands

### Package Management
```bash
# Install dependencies
uv pip install -e .

# List installed packages
uv pip list

# Add new dependencies (edit pyproject.toml then run)
uv pip install -e .
```

### Running the Server
```bash
# The MCP server can be run via the MCP CLI (once implemented)
# Currently, server.py defines the MCP instance but needs implementation
```

## Architecture

The project follows a minimal MCP server structure:

- `server.py`: Main MCP server definition using FastMCP. This is where MCP tools, resources, and prompts should be implemented.
- `main.py`: Entry point file (currently unused, may be removed or repurposed)
- `pyproject.toml`: Project configuration and dependencies

## MCP Server Implementation

The MCP server in `server.py` is initialized but not yet implemented. To add functionality:

1. Define tools using `@mcp.tool()` decorator
2. Define resources using `@mcp.resource()` decorator  
3. Define prompts using `@mcp.prompt()` decorator

Example structure for adding MCP tools:
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Memory")

@mcp.tool()
async def tool_name(param: str) -> str:
    """Tool description"""
    # Implementation
    return result
```

## Current Status

The project is in initial setup phase. The MCP server framework is installed but no memory-related functionality has been implemented yet.

## Memory System Design

A comprehensive design has been created for the memory system. See `MEMORY_DESIGN.md` for full details.

Key design decisions:
- SQLite storage with binary embeddings (1536-dim vectors)
- Multi-agent support with separate memory stores per agent
- Semantic search with recency and context weighting
- Automatic context capture from environment and content
- Memory decay and strengthening based on access patterns

Implementation approach:
1. Start with `initialize_agent`, `observe`, and `recall` tools
2. Use OpenAI text-embedding-3-small for semantic search
3. Store memories in SQLite with BLOB embedding storage
4. Implement relevance scoring combining semantic similarity, recency, and context
5. Add memory pruning when approaching 50K memories per agent