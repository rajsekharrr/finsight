"""
Simple JSON file-based cache with TTL.
Prevents repeated API calls for the same ticker within TTL window.
"""
import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache")
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "6"))


def _cache_path(key: str) -> Path:
    """Sanitize key and return cache file path."""
    # Remove unsafe characters and replace with underscore
    safe_key = key.replace("/", "_").replace("\\", "_").replace(".", "_").replace(" ", "_").replace(":", "_")
    # Limit length to avoid filesystem issues
    safe_key = safe_key[:200]
    return CACHE_DIR / f"{safe_key}.json"


def cache_get(key: str) -> Optional[Any]:
    """
    Return cached value if it exists and is not expired, else None.
    Never crashes on corrupted cache files - returns None and logs warning.
    """
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Check if data has required structure
        if not isinstance(data, dict) or "_ts" not in data:
            logger.warning(f"Cache file corrupted (missing _ts) for {key}, ignoring")
            return None

        age_hours = (time.time() - data["_ts"]) / 3600
        if age_hours > CACHE_TTL_HOURS:
            logger.debug(f"Cache expired for {key} (age: {age_hours:.1f}h)")
            return None

        logger.debug(f"Cache hit for {key}")
        return data.get("value")

    except json.JSONDecodeError as e:
        logger.warning(f"Cache file corrupted (JSON decode error) for {key}: {e}")
        return None
    except Exception as e:
        logger.warning(f"Cache read error for {key}: {e}")
        return None


def cache_set(key: str, value: Any) -> None:
    """Write value to cache with current timestamp."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(key)

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({"_ts": time.time(), "value": value}, f)
        logger.debug(f"Cached {key}")
    except Exception as e:
        logger.warning(f"Cache write error for {key}: {e}")


def cached(key_prefix: str):
    """
    Decorator: cache the return value of a function.
    Uses first positional argument as part of cache key.
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # Build cache key from prefix and first arg if available
            key = f"{key_prefix}_{args[0]}" if args else key_prefix

            # Try to get from cache
            result = cache_get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = fn(*args, **kwargs)
            cache_set(key, result)
            return result

        return wrapper
    return decorator
