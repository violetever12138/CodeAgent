# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

import asyncio
import random
import time
import traceback
from functools import wraps
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry_with(
    func: Callable[..., T],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., T]:
    """
    Decorator that adds retry logic with randomized backoff.

    Args:
        func: The function to decorate
        provider_name: The name of the model provider being called
        max_retries: Maximum number of retry attempts

    Returns:
        Decorated function with retry logic
    """

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        last_exception = None

        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt == max_retries:
                    # Last attempt, re-raise the exception
                    raise

                sleep_time = random.randint(3, 30)
                this_error_message = str(e)
                print(
                    f"{provider_name} API call failed: {this_error_message}. Will sleep for {sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )
                # Randomly sleep for 3-30 seconds
                time.sleep(sleep_time)

        # This should never be reached, but just in case
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper


def async_retry_with(
    func: Callable[..., Any],
    provider_name: str = "OpenAI",
    max_retries: int = 3,
) -> Callable[..., Any]:
    """Cancellation-aware retry wrapper for native async provider calls."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        last_exception: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exception = exc
                if attempt == max_retries:
                    raise
                sleep_time = random.randint(3, 30)
                print(
                    f"{provider_name} API call failed: {exc}. Will sleep for "
                    f"{sleep_time} seconds and will retry.\n{traceback.format_exc()}"
                )
                await asyncio.sleep(sleep_time)
        raise last_exception or Exception("Retry failed for unknown reason")

    return wrapper
