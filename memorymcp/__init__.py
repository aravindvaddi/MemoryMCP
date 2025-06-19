"""MemoryMCP - A Model Context Protocol server for persistent memory and learning"""

from .server import mcp

def main():
    """Main entry point for the MCP server"""
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def run():
        async with stdio_server() as (read_stream, write_stream):
            await mcp.run(
                read_stream=read_stream,
                write_stream=write_stream,
                initialization_options={}
            )
    
    asyncio.run(run())

__all__ = ['mcp', 'main']