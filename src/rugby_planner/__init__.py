from __future__ import annotations

try:
	from dotenv import load_dotenv
	load_dotenv()
except Exception:
	pass

# Basic logging config if none is set by the host process
import logging, os
if not logging.getLogger().handlers:
	level = logging.DEBUG if os.getenv("RUGBY_DEBUG") else logging.INFO
	logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# Reduce noisy client HTTP logs in CLI (stop printing API calls)
_httpx_level = os.getenv("HTTPX_LOG_LEVEL", "WARNING").upper()
try:
	logging.getLogger("httpx").setLevel(getattr(logging, _httpx_level, logging.WARNING))
	logging.getLogger("httpcore").setLevel(getattr(logging, _httpx_level, logging.WARNING))
except Exception:
	# Fallback: ensure warnings-and-above only
	logging.getLogger("httpx").setLevel(logging.WARNING)
	logging.getLogger("httpcore").setLevel(logging.WARNING)

__all__ = ["main"]
