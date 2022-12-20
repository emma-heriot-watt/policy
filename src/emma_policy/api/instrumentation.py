from opentelemetry import trace

from emma_policy._version import __version__  # noqa: WPS436


def get_tracer(name: str) -> trace.Tracer:
    """Get the tracer."""
    return trace.get_tracer(name, __version__)
