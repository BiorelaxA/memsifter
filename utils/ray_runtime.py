"""Lightweight initialization helpers for Ray worker processes."""

import asyncio


def setup_worker_event_loop():
    """Provide the current loop required by Ray 2.48's async Data UDFs.

    Ray installs uvloop in its workers when that package is available.
    Unlike the Python 3.11 default policy, uvloop's policy does not create a
    main-thread loop implicitly.  Preserve the installed policy and create a
    loop only when the worker does not already have one.
    """
    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass

    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        loop = None

    if loop is None or loop.is_closed():
        asyncio.set_event_loop(policy.new_event_loop())
