import os
import logging
import threading
import queue

LOG = logging.getLogger(__name__)

# Lightweight wrapper for sending events to Inngest. If the real `inngest` SDK
# is installed and configured (via INNGEST_API_KEY), this will try to use it.
# Otherwise it falls back to a local background worker that imports the
# process_document handler and runs it in a thread. This keeps upload fast
# while enabling a simple local workflow during development.

_local_queue = queue.Queue()


def _local_worker():
    from inngest.functions import process_document as proc  # type: ignore

    while True:
        event_name, payload = _local_queue.get()
        try:
            LOG.info("[local worker] processing event %s", event_name)
            proc.handle_event(payload)
        except Exception:
            LOG.exception("[local worker] event handler failed")
        finally:
            _local_queue.task_done()


_worker_thread = threading.Thread(target=_local_worker, daemon=True)
_worker_thread.start()


def send_event(event_name: str, payload: dict):
    """Send an event to Inngest or enqueue locally as a fallback.

    event_name: e.g. 'document/uploaded'
    payload: event payload dictionary
    """
    # Try to use the official SDK if configured
    try:
        import inngest  # type: ignore

        api_key = os.getenv("INNGEST_API_KEY")
        if api_key:
            # Use the SDK's client if available. We keep this generic because
            # exact SDK usage may vary by version; sending via local queue is
            # a safe fallback in development.
            try:
                client = inngest.Inngest(api_key=api_key)  # type: ignore
                client.send(event_name, payload)  # type: ignore
                LOG.info("Sent event to Inngest SDK: %s", event_name)
                return
            except Exception:
                LOG.exception("Failed to send via inngest SDK, falling back to local queue")

    except Exception:
        # SDK not installed or import failed; fall back below
        pass

    # Local enqueue fallback
    LOG.info("Enqueuing event locally: %s", event_name)
    _local_queue.put((event_name, payload))
