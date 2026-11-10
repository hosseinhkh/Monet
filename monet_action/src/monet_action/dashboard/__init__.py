"""Monitoring dashboard for Monet-Action training runs and evaluations (stdlib only).

Run ``python -m monet_action.dashboard --runs runs --results results`` and open the printed URL.
See ``docs/DASHBOARD.md`` for the file contract, API and security notes.
"""
from .server import VERSION, main, make_server  # noqa: F401
