#!/usr/bin/env python3
"""
Memory MCP Server Entry Point

This runs the Memory MCP server that provides multi-agent memory management.
"""

import asyncio
from server import mcp


def main():
    """Run the Memory MCP server"""
    print("Starting Memory MCP Server...")
    print("This server provides memory management for multiple AI agents.")
    print("Each agent maintains its own persistent memory store.")
    
    # Run the MCP server
    asyncio.run(mcp.run())


if __name__ == "__main__":
    main()
