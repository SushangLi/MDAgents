"""
File operation utilities for reading, writing, and managing files.
"""

import os
from pathlib import Path
from typing import List, Optional

from utils.logging import get_logger

logger = get_logger(__name__)


def read_file(path: str) -> str:
    """
    Read file contents.

    Args:
        path: Path to file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
    """
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Read file: {path} ({len(content)} chars)")
        return content
    except UnicodeDecodeError:
        # Try binary mode for non-text files
        with open(file_path, "rb") as f:
            content = f.read()
        logger.debug(f"Read binary file: {path} ({len(content)} bytes)")
        return f"<binary file: {len(content)} bytes>"
    except Exception as e:
        logger.error(f"Error reading file {path}: {e}")
        raise


def write_file(path: str, content: str) -> bool:
    """
    Write content to file.

    Args:
        path: Path to file
        content: Content to write

    Returns:
        True if successful

    Raises:
        PermissionError: If file can't be written
    """
    file_path = Path(path)

    # Create parent directories if needed
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.debug(f"Wrote file: {path} ({len(content)} chars)")
        return True
    except Exception as e:
        logger.error(f"Error writing file {path}: {e}")
        raise


def list_directory(dir_path: str, pattern: Optional[str] = None) -> List[str]:
    """
    List directory contents.

    Args:
        dir_path: Path to directory
        pattern: Optional glob pattern to filter files (e.g., "*.py")

    Returns:
        List of file/directory names

    Raises:
        FileNotFoundError: If directory doesn't exist
        NotADirectoryError: If path is not a directory
    """
    path = Path(dir_path)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")

    try:
        if pattern:
            items = [str(p.relative_to(path)) for p in path.glob(pattern)]
        else:
            items = [p.name for p in path.iterdir()]

        logger.debug(f"Listed directory: {dir_path} ({len(items)} items)")
        return sorted(items)
    except Exception as e:
        logger.error(f"Error listing directory {dir_path}: {e}")
        raise


def get_file_info(file_path: str) -> dict:
    """
    Get file metadata.

    Args:
        file_path: Path to file

    Returns:
        Dict with file metadata

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    stat = path.stat()

    return {
        "name": path.name,
        "path": str(path.absolute()),
        "size": stat.st_size,
        "is_file": path.is_file(),
        "is_directory": path.is_dir(),
        "created": stat.st_ctime,
        "modified": stat.st_mtime,
        "extension": path.suffix,
    }


def search_files(root_dir: str, pattern: str) -> List[str]:
    """
    Search for files matching pattern recursively.

    Args:
        root_dir: Root directory to search from
        pattern: Glob pattern (e.g., "**/*.py", "*.csv")

    Returns:
        List of matching file paths

    Raises:
        FileNotFoundError: If root directory doesn't exist
    """
    path = Path(root_dir)

    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {root_dir}")

    try:
        matches = [str(p) for p in path.glob(pattern)]
        logger.debug(f"Found {len(matches)} files matching '{pattern}' in {root_dir}")
        return sorted(matches)
    except Exception as e:
        logger.error(f"Error searching files in {root_dir}: {e}")
        raise


def file_exists(file_path: str) -> bool:
    """
    Check if file exists.

    Args:
        file_path: Path to file

    Returns:
        True if file exists, False otherwise
    """
    return Path(file_path).exists()


def delete_file(file_path: str) -> bool:
    """
    Delete file.

    Args:
        file_path: Path to file

    Returns:
        True if successful

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be deleted
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        path.unlink()
        logger.debug(f"Deleted file: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error deleting file {file_path}: {e}")
        raise
