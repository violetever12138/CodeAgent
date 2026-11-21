"""Optional request observer; ordinary Trae calls retain their behavior."""

from contextlib import nullcontext
from contextvars import ContextVar

observer = ContextVar("request_observer", default=None)
last_request = ContextVar("last_observed_request", default=None)
logical_call = ContextVar("observed_logical_call", default=None)


class NoRequest:
    def response(self, value):
        pass


def request(provider, payload):
    current = observer.get()
    return current.request(provider, payload) if current else nullcontext(NoRequest())


def sdk_client(client):
    # Only the observed CodeSwarm path owns retry accounting. Disable invisible
    # SDK retries there; the explicit provider/runtime retry loops remain.
    if observer.get() and isinstance(getattr(client, "max_retries", None), int):
        return client.with_options(max_retries=0)
    return client


def google_config(config):
    if observer.get():
        from google.genai import types
        options = config.http_options.model_copy(deep=True) if config.http_options else types.HttpOptions()
        options.retry_options = types.HttpRetryOptions(attempts=1)
        return config.model_copy(update={"http_options": options})
    return config
