import atexit
import asyncio
import threading
import warnings

from hindsight_client import Hindsight

from src.config import (
    HINDSIGHT_API_KEY,
    HINDSIGHT_BASE_URL
)


_loop = None
_loop_thread = None
_client = None
_loop_lock = threading.Lock()


def _get_loop():
    global _client, _loop, _loop_thread

    with _loop_lock:
        if (
            _loop is not None
            and not _loop.is_closed()
            and _loop.is_running()
        ):
            return _loop

        # A client owns an aiohttp session bound to its original loop.
        # Never reuse it if Streamlit or shutdown stopped that loop.
        _client = None
        ready = threading.Event()

        def run_loop():
            global _loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _loop = loop
            loop.call_soon(ready.set)
            loop.run_forever()

        _loop_thread = threading.Thread(
            target=run_loop,
            name="hindsight-event-loop",
            daemon=True
        )
        _loop_thread.start()

        if not ready.wait(timeout=5):
            raise RuntimeError("Hindsight event loop failed to start.")

        if _loop is None or _loop.is_closed() or not _loop.is_running():
            raise RuntimeError("Hindsight event loop stopped during startup.")

        return _loop


def _run_with_client(operation):
    async def run():
        global _client

        if _client is None:
            _client = Hindsight(
                base_url=HINDSIGHT_BASE_URL,
                api_key=HINDSIGHT_API_KEY
            )

        return await operation(_client)

    future = asyncio.run_coroutine_threadsafe(run(), _get_loop())
    return future.result()


def _shutdown():
    global _client, _loop, _loop_thread

    loop = _loop
    if loop is None or loop.is_closed() or not loop.is_running():
        return

    async def close_client():
        global _client
        if _client is not None:
            await _client.aclose()
            _client = None

    future = asyncio.run_coroutine_threadsafe(close_client(), loop)
    try:
        future.result(timeout=10)
    except Exception as error:
        warnings.warn(
            f"Could not close the Hindsight client cleanly: {error}",
            RuntimeWarning
        )
    finally:
        loop.call_soon_threadsafe(loop.stop)
        if _loop_thread is not None:
            _loop_thread.join(timeout=10)
        if not loop.is_running() and not loop.is_closed():
            loop.close()
        _client = None
        _loop = None
        _loop_thread = None


atexit.register(_shutdown)


def get_bank_id(customer_id):
    # Keep bank IDs predictable and separate per customer.
    safe_id = "".join(
        c if c.isalnum() or c in "-_" else "-"
        for c in customer_id.lower()
    )
    return f"smartcare-{safe_id[:50]}"


async def _initialize_customer(client, customer_id):
    bank_id = get_bank_id(customer_id)

    try:
        await client.acreate_bank(
            bank_id=bank_id,
            name=f"SmartCare Customer {customer_id}",
            background=(
                "Long-term memory for a customer support agent. "
                "Remember relevant customer environment, recurring "
                "issues, preferences, and confirmed troubleshooting outcomes."
            )
        )
    except Exception as error:
        # The bank may already exist. Do not silently hide other errors.
        message = str(error).lower()
        if "already exists" not in message and "409" not in message:
            raise

    return bank_id


def initialize_customer(customer_id):
    return _run_with_client(
        lambda client: _initialize_customer(client, customer_id)
    )


def recall_memory(customer_id, question):
    async def recall(client):
        bank_id = await _initialize_customer(client, customer_id)

        result = await client.arecall(
            bank_id=bank_id,
            query=question,
            budget="low"
        )

        memories = []

        for item in getattr(result, "results", []):
            text = getattr(item, "text", None)
            if text:
                memories.append(text)

        return memories

    return _run_with_client(recall)


def store_memory(customer_id, memory_text):
    if not memory_text or not memory_text.strip():
        return

    async def store(client):
        bank_id = await _initialize_customer(client, customer_id)

        await client.aretain(
            bank_id=bank_id,
            content=memory_text.strip(),
            context="SmartCare customer support conversation"
        )

    _run_with_client(store)