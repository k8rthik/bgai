"""Crawler identity: the contact address every outbound request carries.

terra.snellman.net's ``/app/`` endpoints are robots-disallowed, so this
project only fetches from them politely -- at most 1 request/second, with
a User-Agent that names the project and a real contact address. The
address is deliberately NOT in source: it is read from the
``BGAI_CRAWL_CONTACT`` environment variable (a ``.env`` file in the working
directory or any parent is loaded if present; see ``.env.example``). Requests fail fast with a clear
message when it is unset, so an anonymous crawl cannot happen by accident.
"""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv

CONTACT_ENV_VAR = "BGAI_CRAWL_CONTACT"
PROJECT_TAG = "bgai-research/0.1"
PURPOSE = "Terra Mystica AI research"


class MissingCrawlContactError(RuntimeError):
    """Raised when an outbound crawl request would carry no contact address."""


def crawl_contact() -> str:
    """Return the contact address for outbound requests, or raise.

    ``load_dotenv`` is a no-op when no ``.env`` exists and never overrides a
    variable that is already set, so an explicit environment always wins.
    """
    load_dotenv(find_dotenv(usecwd=True))
    contact = os.environ.get(CONTACT_ENV_VAR, "").strip()
    if not contact:
        raise MissingCrawlContactError(
            f"{CONTACT_ENV_VAR} is not set. The crawler identifies itself to "
            "terra.snellman.net and tmtour.org with a contact address; set it "
            "in the environment or in a .env at the repo root (see .env.example)."
        )
    return contact


def user_agent() -> str:
    """The User-Agent header for every request this project sends."""
    return f"{PROJECT_TAG} ({PURPOSE}; contact: {crawl_contact()})"
