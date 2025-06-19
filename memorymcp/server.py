from mcp.server.fastmcp import FastMCP
from mcp.types import Tool, TextContent
from pydantic import Field
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
import sqlite3
from contextlib import closing
import platform
import socket
import re
import subprocess
import struct
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create an MCP server
mcp = FastMCP("Memory")

# Base directory for all agent memory stores
MEMORY_BASE_DIR = Path.home() / ".memorymcp" / "agents"

# Initialize embedding model (using a small, efficient model)
embedding_model = None
def get_embedding_model():
    global embedding_model
    if embedding_model is None:
        # Using all-MiniLM-L6-v2: small, fast, good quality
        # 384 dimensions, ~80MB download
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return embedding_model

class MemoryStore:
    """SQLite-based memory store with semantic search capabilities"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.db_path = MEMORY_BASE_DIR / agent_id / "memories.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Initialize the database schema matching MEMORY_DESIGN.md"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            # Check if we need to migrate from old schema
            cursor = conn.execute("PRAGMA table_info(memories)")
            columns = {row[1] for row in cursor.fetchall()}
            
            if columns and "embedding" not in columns:
                # Migration needed - rename old table
                conn.execute("ALTER TABLE memories RENAME TO memories_old")
            
            # Create new schema
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    strength REAL DEFAULT 1.0,
                    access_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    last_accessed DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context JSON,
                    metadata JSON
                )
            """)
            
            # Create indexes
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_id ON memories(agent_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_strength ON memories(strength)")
            
            # Migrate old data if exists
            if "memories_old" in [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]:
                # Note: Old memories won't have embeddings, so they'll be less useful
                conn.execute("""
                    INSERT INTO memories (agent_id, content, embedding, created_at, metadata)
                    SELECT ?, content, X'00', timestamp, metadata
                    FROM memories_old
                """, (self.agent_id,))
                conn.execute("DROP TABLE memories_old")
            
            conn.commit()
    
    async def add_memory(self, content: str, embedding: np.ndarray, context: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Add a new memory with embedding to the store"""
        # Convert numpy array to binary blob
        embedding_blob = struct.pack('f' * 384, *embedding.tolist())
        
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.execute("""
                INSERT INTO memories (agent_id, content, embedding, strength, context, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                self.agent_id,
                content,
                embedding_blob,
                metadata.get("importance", 1.0) if metadata else 1.0,
                json.dumps(context or {}),
                json.dumps(metadata or {})
            ))
            conn.commit()
            return cursor.lastrowid
    
    async def search_memories(self, query_embedding: np.ndarray, limit: int = 10, recency_weight: float = 0.3) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity with relevance scoring"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get all memories with embeddings
            cursor = conn.execute("""
                SELECT id, content, embedding, strength, access_count, 
                       created_at, last_accessed, context, metadata
                FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT 1000
            """, (self.agent_id,))
            
            memories = []
            current_time = datetime.utcnow()
            
            for row in cursor:
                # Deserialize embedding
                embedding_blob = row['embedding']
                if len(embedding_blob) == 1:  # Skip placeholder embeddings
                    continue
                    
                memory_embedding = np.array(struct.unpack('f' * 384, embedding_blob))
                
                # Calculate semantic similarity
                semantic_similarity = cosine_similarity(
                    query_embedding.reshape(1, -1), 
                    memory_embedding.reshape(1, -1)
                )[0][0]
                
                # Calculate recency score (exponential decay)
                created_at = datetime.fromisoformat(row['created_at'])
                days_old = (current_time - created_at).days
                recency_score = np.exp(-days_old / 30)  # 50% decay after 30 days
                
                # Calculate final score
                final_score = (semantic_similarity * (1 - recency_weight) + 
                              recency_score * recency_weight) * row['strength']
                
                memories.append({
                    'id': row['id'],
                    'content': row['content'],
                    'score': final_score,
                    'semantic_similarity': semantic_similarity,
                    'recency_score': recency_score,
                    'strength': row['strength'],
                    'access_count': row['access_count'],
                    'created_at': row['created_at'],
                    'context': json.loads(row['context'] or '{}'),
                    'metadata': json.loads(row['metadata'] or '{}')
                })
                
                # Update access count and last accessed
                conn.execute("""
                    UPDATE memories 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP,
                        strength = MIN(strength * 1.1, 1.0)
                    WHERE id = ?
                """, (row['id'],))
            
            conn.commit()
            
            # Sort by score and return top results
            memories.sort(key=lambda x: x['score'], reverse=True)
            return memories[:limit]
    
    def get_recent_memories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent memories"""
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT id, content, strength, access_count, 
                       created_at, last_accessed, context, metadata
                FROM memories
                WHERE agent_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (self.agent_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]

# Global registry of active memory stores
memory_stores: Dict[str, MemoryStore] = {}

def get_or_create_memory_store(agent_id: str) -> MemoryStore:
    """Get existing memory store or create a new one for an agent"""
    if agent_id not in memory_stores:
        memory_stores[agent_id] = MemoryStore(agent_id)
    return memory_stores[agent_id]

@mcp.tool()
def initialize_agent(
    agent_id: str = Field(description="Unique identifier for the agent instance"),
    personality: Optional[str] = Field(
        None, 
        description="Optional personality description or system prompt for the agent"
    ),
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata about the agent (e.g., capabilities, preferences)"
    )
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
        "metadata": metadata or {}
    }
    
    if config_path.exists():
        # Load existing config
        with open(config_path, 'r') as f:
            existing_config = json.load(f)
            config["created_at"] = existing_config.get("created_at", config["created_at"])
            config["first_connected"] = existing_config.get("first_connected", config["created_at"])
        config["last_connected"] = datetime.utcnow().isoformat()
    else:
        config["first_connected"] = config["created_at"]
        config["last_connected"] = config["created_at"]
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize memory store
    store = get_or_create_memory_store(agent_id)
    
    return f"Agent '{agent_id}' initialized. Memory store ready at {agent_dir}"

@mcp.tool()
async def observe(
    agent_id: str = Field(description="The agent instance making the observation"),
    content: str = Field(description="The observation or experience to remember"),
    importance: Optional[float] = Field(
        None, 
        description="Importance score (0-1) for the memory",
        ge=0, 
        le=1
    ),
    tags: Optional[List[str]] = Field(
        None,
        description="Tags to categorize the memory"
    ),
    context: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional context about the observation"
    )
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
    
    metadata = {
        "importance": importance or 0.5,
        "tags": tags or [],
        "type": "observation"
    }
    
    memory_id = await store.add_memory(content, embedding, auto_context, metadata)
    
    return f"Memory #{memory_id} stored for agent '{agent_id}'"

@mcp.tool()
async def recall(
    agent_id: str = Field(description="The agent instance recalling memories"),
    query: Optional[str] = Field(
        None,
        description="Search query to find relevant memories"
    ),
    limit: int = Field(
        10,
        description="Maximum number of memories to recall",
        ge=1,
        le=50
    ),
    recency_bias: bool = Field(
        True,
        description="Whether to prioritize recent memories"
    )
) -> List[Dict[str, Any]]:
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
        except Exception as e:
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
        if isinstance(memory, dict) and 'metadata' in memory and isinstance(memory['metadata'], dict):
            # From semantic search
            formatted_memories.append({
                "id": memory["id"],
                "timestamp": memory.get("created_at", memory.get("timestamp", "")),
                "content": memory["content"],
                "importance": memory["metadata"].get("importance", 0.5),
                "tags": memory["metadata"].get("tags", []),
                "context": memory.get("context", {}),
                "score": memory.get("score", 0.0)
            })
        else:
            # From get_recent_memories
            metadata = json.loads(memory.get("metadata", "{}")) if "metadata" in memory else {}
            formatted_memories.append({
                "id": memory["id"],
                "timestamp": memory.get("created_at", memory.get("timestamp", "")),
                "content": memory["content"],
                "importance": metadata.get("importance", 0.5),
                "tags": metadata.get("tags", []),
                "context": json.loads(memory.get("context", "{}")) if "context" in memory else {},
                "score": 0.0
            })
    
    return formatted_memories

@mcp.tool()
async def reflect(
    agent_id: str = Field(description="The agent instance performing reflection"),
    topic: str = Field(description="Topic or theme to reflect on"),
    depth: int = Field(
        20,
        description="Number of memories to consider for reflection",
        ge=5,
        le=100
    )
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
    reflection_parts = [
        f"Reflection on '{topic}' based on {len(memories)} memories:",
        ""
    ]
    
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
    reflection_content = f"Reflected on '{topic}': Found {len(memories)} relevant memories with patterns in {list(tag_groups.keys())}"
    
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
        "importance": 0.8  # Reflections are important
    }
    
    await store.add_memory(
        reflection_content, 
        reflection_embedding,
        capture_system_context(),
        reflection_metadata
    )
    
    reflection_parts.append("")
    reflection_parts.append("Reflection stored as new memory for future reference.")
    
    return "\n".join(reflection_parts)

@mcp.tool()
def list_agents() -> List[Dict[str, Any]]:
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
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    agents.append({
                        "agent_id": config["agent_id"],
                        "created_at": config.get("created_at"),
                        "last_connected": config.get("last_connected"),
                        "personality": config.get("personality"),
                        "metadata": config.get("metadata", {})
                    })
    
    return sorted(agents, key=lambda x: x.get("last_connected", ""), reverse=True)

@mcp.tool()
def export_memories(
    agent_id: str = Field(description="The agent whose memories to export"),
    format: str = Field(
        "json",
        description="Export format: 'json' or 'markdown'",
        pattern="^(json|markdown)$"
    ),
    include_metadata: bool = Field(
        True,
        description="Whether to include metadata in the export"
    )
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
            "memories": memories if include_metadata else [
                {"content": m["content"], "timestamp": m["timestamp"]} 
                for m in memories
            ]
        }
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
    else:  # markdown
        export_path = export_dir / f"memories_{timestamp}.md"
        with open(export_path, 'w') as f:
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

# Context capture utilities
def capture_system_context() -> Dict[str, Any]:
    """Capture available system context automatically."""
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "cwd": os.getcwd(),
        "platform": {
            "system": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        },
        "environment": {
            "user": os.environ.get("USER", "unknown"),
            "home": str(Path.home()),
            "shell": os.environ.get("SHELL", "unknown")
        }
    }
    
    # Capture development environment indicators
    dev_env_vars = ["VIRTUAL_ENV", "CONDA_DEFAULT_ENV", "NODE_ENV", "PYTHONPATH", "GOPATH"]
    dev_context = {}
    for var in dev_env_vars:
        if var in os.environ:
            dev_context[var] = os.environ[var]
    
    if dev_context:
        context["development_env"] = dev_context
    
    # Try to detect project context from cwd
    cwd_path = Path(os.getcwd())
    project_indicators = [".git", "package.json", "pyproject.toml", "Cargo.toml", "go.mod"]
    
    for indicator in project_indicators:
        if (cwd_path / indicator).exists():
            context["project_type"] = indicator
            context["project_name"] = cwd_path.name
            break
    
    return context

def extract_file_references(content: str) -> List[str]:
    """Extract file paths mentioned in content."""
    # Pattern matching for file paths
    patterns = [
        r'(?:^|[\s"])(/[\w\-./]+\.\w+)',  # Absolute paths
        r'(?:^|[\s"])(\./[\w\-./]+\.\w+)',  # Relative paths starting with ./
        r'(?:^|[\s"])([\w\-]+\.\w+)',  # Simple filenames
        r'`([^`]+\.\w+)`',  # Files in backticks
        r'"([^"]+\.\w+)"',  # Files in quotes
        r'\'([^\']+\.\w+)\'',  # Files in single quotes
    ]
    
    files = []
    for pattern in patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.IGNORECASE)
        files.extend(matches)
    
    # Deduplicate and validate
    valid_files = []
    for f in set(files):
        try:
            # Basic validation - has extension and reasonable length
            if '.' in f and len(f) < 256:
                valid_files.append(f)
        except:
            pass
    
    return valid_files

def infer_recent_files() -> List[Dict[str, Any]]:
    """
    Infer recently accessed files from various sources.
    Note: This is a best-effort approach since MCP servers can't monitor filesystem.
    """
    recent_files = []
    
    # Check common project files in cwd
    cwd = Path.cwd()
    common_patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go", "*.rs", "*.md"]
    
    for pattern in common_patterns:
        try:
            for file in cwd.glob(pattern):
                if file.is_file():
                    stat = file.stat()
                    recent_files.append({
                        "path": str(file),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "size": stat.st_size
                    })
        except:
            pass
    
    # Sort by modification time
    recent_files.sort(key=lambda x: x["modified"], reverse=True)
    
    # Return top 10 most recently modified
    return recent_files[:10]

def analyze_content_for_context(content: str) -> Dict[str, Any]:
    """Analyze content to suggest relevant context."""
    context_hints = {
        "detected_languages": [],
        "detected_frameworks": [],
        "detected_concepts": [],
        "file_references": extract_file_references(content)
    }
    
    content_lower = content.lower()
    
    # Language detection
    language_indicators = {
        "python": ["import ", "def ", "class ", "__init__", "self.", "pip ", "requirements.txt"],
        "javascript": ["const ", "let ", "var ", "function ", "=>", "npm ", "package.json"],
        "typescript": ["interface ", "type ", ": string", ": number", "tsconfig.json"],
        "java": ["public class", "private ", "public static", "import java", "maven", "gradle"],
        "go": ["func ", "package ", "import (", "go mod"],
        "rust": ["fn ", "let mut", "impl ", "cargo.toml", "use std"],
    }
    
    for lang, indicators in language_indicators.items():
        if any(ind in content_lower for ind in indicators):
            context_hints["detected_languages"].append(lang)
    
    # Framework detection
    framework_indicators = {
        "react": ["react", "usestate", "useeffect", "jsx", "component"],
        "vue": ["vue", "v-if", "v-for", "@click", "mounted()"],
        "django": ["django", "models.py", "views.py", "urls.py", "settings.py"],
        "fastapi": ["fastapi", "@app.get", "@app.post", "pydantic"],
        "express": ["express", "app.get", "app.post", "req, res"],
        "flask": ["flask", "@app.route", "render_template"],
    }
    
    for framework, indicators in framework_indicators.items():
        if any(ind in content_lower for ind in indicators):
            context_hints["detected_frameworks"].append(framework)
    
    # Concept detection
    concept_indicators = {
        "api": ["api", "endpoint", "rest", "graphql", "request", "response"],
        "database": ["database", "sql", "query", "table", "schema", "migration"],
        "testing": ["test", "assert", "expect", "describe", "it(", "jest", "pytest"],
        "debugging": ["error", "bug", "fix", "issue", "problem", "traceback"],
        "documentation": ["docs", "readme", "comment", "docstring", "jsdoc"],
    }
    
    for concept, indicators in concept_indicators.items():
        if any(ind in content_lower for ind in indicators):
            context_hints["detected_concepts"].append(concept)
    
    # Remove duplicates
    for key in ["detected_languages", "detected_frameworks", "detected_concepts"]:
        context_hints[key] = list(set(context_hints[key]))
    
    return context_hints

async def observe_with_context_internal(
    agent_id: str = Field(description="The agent instance making the observation"),
    content: str = Field(description="The observation or experience to remember"),
    importance: Optional[float] = Field(
        None, 
        description="Importance score (0-1) for the memory",
        ge=0, 
        le=1
    ),
    tags: Optional[List[str]] = Field(
        None,
        description="Tags to categorize the memory"
    ),
    explicit_context: Optional[Dict[str, Any]] = Field(
        None,
        description="Explicitly provided context (e.g., current task, project info)"
    ),
    capture_system: bool = Field(
        True,
        description="Whether to automatically capture system context"
    ),
    related_files: Optional[List[str]] = Field(
        None,
        description="List of files related to this memory"
    )
) -> Dict[str, Any]:
    """
    Enhanced observation storage with automatic context capture.
    Captures system context, analyzes content, and stores rich metadata.
    """
    store = get_or_create_memory_store(agent_id)
    
    # Build comprehensive context
    context = {}
    
    # 1. System context
    if capture_system:
        context["system"] = capture_system_context()
    
    # 2. Explicit context from caller
    if explicit_context:
        context["explicit"] = explicit_context
    
    # 3. Analyzed context from content
    context["inferred"] = analyze_content_for_context(content)
    
    # 4. File context
    if related_files:
        context["related_files"] = related_files
    else:
        # Use detected files from content analysis
        context["related_files"] = context["inferred"].get("file_references", [])
    
    # 5. Try to get recent files (best effort)
    if capture_system:
        context["recent_files"] = infer_recent_files()
    
    # Build metadata
    metadata = {
        "importance": importance or 0.5,
        "tags": tags or [],
        "type": "observation",
        "context": context,
        "context_version": "1.0"  # For future compatibility
    }
    
    # Auto-generate tags from analysis
    auto_tags = []
    if context["inferred"]["detected_languages"]:
        auto_tags.extend(context["inferred"]["detected_languages"])
    if context["inferred"]["detected_frameworks"]:
        auto_tags.extend(context["inferred"]["detected_frameworks"])
    
    # Merge with provided tags
    all_tags = list(set((tags or []) + auto_tags))
    metadata["tags"] = all_tags
    
    # Store the memory
    memory_id = store.add_memory(content, metadata)
    
    # Return rich response
    return {
        "memory_id": memory_id,
        "agent_id": agent_id,
        "timestamp": datetime.utcnow().isoformat(),
        "context_captured": {
            "system": bool(context.get("system")),
            "explicit": bool(context.get("explicit")),
            "inferred_languages": context["inferred"]["detected_languages"],
            "inferred_frameworks": context["inferred"]["detected_frameworks"],
            "file_references": len(context.get("related_files", [])),
            "recent_files": len(context.get("recent_files", [])) if "recent_files" in context else 0
        },
        "tags": all_tags
    }

@mcp.tool()
def search_by_context(
    agent_id: str = Field(description="The agent instance searching memories"),
    context_filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Context filters (e.g., {'system.cwd': '/path', 'inferred.detected_languages': ['python']})"
    ),
    text_query: Optional[str] = Field(
        None,
        description="Optional text search within matching memories"
    ),
    limit: int = Field(
        10,
        description="Maximum number of results",
        ge=1,
        le=100
    )
) -> List[Dict[str, Any]]:
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
        filtered_memories.append({
            "id": memory["id"],
            "timestamp": memory["timestamp"],
            "content": memory["content"],
            "metadata": metadata,
            "context_summary": {
                "cwd": context.get("system", {}).get("cwd", "unknown"),
                "languages": context.get("inferred", {}).get("detected_languages", []),
                "frameworks": context.get("inferred", {}).get("detected_frameworks", []),
                "files": len(context.get("related_files", []))
            }
        })
    
    # Sort by timestamp (most recent first)
    filtered_memories.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return filtered_memories[:limit]

@mcp.tool()
def get_context_summary(
    agent_id: str = Field(description="The agent instance to analyze")
) -> Dict[str, Any]:
    """
    Get a summary of all contexts captured for an agent's memories.
    Useful for understanding the agent's knowledge domains and work history.
    """
    store = get_or_create_memory_store(agent_id)
    memories = store.get_recent_memories(limit=1000)
    
    summary = {
        "total_memories": len(memories),
        "contexts": {
            "working_directories": {},
            "languages": {},
            "frameworks": {},
            "concepts": {},
            "file_types": {}
        },
        "time_range": {
            "earliest": None,
            "latest": None
        }
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
            summary["contexts"]["working_directories"][cwd] = \
                summary["contexts"]["working_directories"].get(cwd, 0) + 1
        
        # Count languages, frameworks, concepts
        inferred = context.get("inferred", {})
        for lang in inferred.get("detected_languages", []):
            summary["contexts"]["languages"][lang] = \
                summary["contexts"]["languages"].get(lang, 0) + 1
        
        for framework in inferred.get("detected_frameworks", []):
            summary["contexts"]["frameworks"][framework] = \
                summary["contexts"]["frameworks"].get(framework, 0) + 1
        
        for concept in inferred.get("detected_concepts", []):
            summary["contexts"]["concepts"][concept] = \
                summary["contexts"]["concepts"].get(concept, 0) + 1
        
        # Count file types
        for file_ref in context.get("related_files", []):
            if "." in file_ref:
                ext = file_ref.split(".")[-1].lower()
                summary["contexts"]["file_types"][ext] = \
                    summary["contexts"]["file_types"].get(ext, 0) + 1
    
    # Sort contexts by frequency
    for key in summary["contexts"]:
        summary["contexts"][key] = dict(
            sorted(summary["contexts"][key].items(), 
                   key=lambda x: x[1], 
                   reverse=True)
        )
    
    return summary
