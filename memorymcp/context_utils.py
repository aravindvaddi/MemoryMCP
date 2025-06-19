"""Utilities for capturing and analyzing context from content and environment."""

import os
import platform
import re
from datetime import datetime
from pathlib import Path
from typing import Any


def capture_system_context() -> dict[str, Any]:
    """Capture available system context automatically."""
    context = {
        "timestamp": datetime.utcnow().isoformat(),
        "cwd": os.getcwd(),
        "platform": {
            "system": platform.system(),
            "version": platform.version(),
            "machine": platform.machine(),
            "python_version": platform.python_version(),
        },
        "environment": {
            "user": os.environ.get("USER", "unknown"),
            "home": str(Path.home()),
            "shell": os.environ.get("SHELL", "unknown"),
        },
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


def extract_file_references(content: str) -> list[str]:
    """Extract file paths mentioned in content."""
    # Pattern matching for file paths
    patterns = [
        r'(?:^|[\s"])(/[\w\-./]+\.\w+)',  # Absolute paths
        r'(?:^|[\s"])(\./[\w\-./]+\.\w+)',  # Relative paths starting with ./
        r'(?:^|[\s"])([\w\-]+\.\w+)',  # Simple filenames
        r"`([^`]+\.\w+)`",  # Files in backticks
        r'"([^"]+\.\w+)"',  # Files in quotes
        r"\'([^\']+\.\w+)\'",  # Files in single quotes
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
            if "." in f and len(f) < 256:
                valid_files.append(f)
        except Exception:
            pass

    return valid_files


def infer_recent_files() -> list[dict[str, Any]]:
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
                    recent_files.append(
                        {
                            "path": str(file),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "size": stat.st_size,
                        }
                    )
        except Exception:
            pass

    # Sort by modification time
    recent_files.sort(key=lambda x: x["modified"], reverse=True)

    # Return top 10 most recently modified
    return recent_files[:10]


def analyze_content_for_context(content: str) -> dict[str, Any]:
    """Analyze content to suggest relevant context."""
    context_hints = {
        "detected_languages": [],
        "detected_frameworks": [],
        "detected_concepts": [],
        "file_references": extract_file_references(content),
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

