"""
Rate limiter to prevent overwhelming external RSS feeds.
Uses a simple token bucket algorithm.
"""

import asyncio
import time
from typing import Optional


class RateLimiter:
    """
    Simple rate limiter using token bucket algorithm.
    Ensures we don't make too many requests to external feeds.
    """

    def __init__(self, requests_per_second: float = 2.0):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self._last_request_time: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """
        Wait until we're allowed to make another request.
        Call this before each external request.
        """
        async with self._lock:
            now = time.monotonic()
            time_since_last = now - self._last_request_time

            if time_since_last < self.min_interval:
                wait_time = self.min_interval - time_since_last
                await asyncio.sleep(wait_time)

            self._last_request_time = time.monotonic()

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter(requests_per_second: float = 2.0) -> RateLimiter:
    """Get or create the global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(requests_per_second)
    return _rate_limiter
