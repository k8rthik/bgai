"""The crawler must never send a request without a contact address."""

from __future__ import annotations

import pytest

from bgai.data import identity


def test_user_agent_carries_contact_from_env(monkeypatch) -> None:
    monkeypatch.setenv(identity.CONTACT_ENV_VAR, "  someone@example.org ")
    ua = identity.user_agent()
    assert ua == "bgai-research/0.1 (Terra Mystica AI research; contact: someone@example.org)"


def test_missing_contact_fails_fast_with_a_clear_message(monkeypatch) -> None:
    monkeypatch.delenv(identity.CONTACT_ENV_VAR, raising=False)
    monkeypatch.setattr(identity, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(identity.MissingCrawlContactError, match=identity.CONTACT_ENV_VAR):
        identity.user_agent()


def test_blank_contact_counts_as_missing(monkeypatch) -> None:
    monkeypatch.setenv(identity.CONTACT_ENV_VAR, "   ")
    monkeypatch.setattr(identity, "load_dotenv", lambda *a, **k: False)
    with pytest.raises(identity.MissingCrawlContactError):
        identity.crawl_contact()


def test_explicit_environment_wins_over_dotenv(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(f"{identity.CONTACT_ENV_VAR}=dotenv@example.org\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(identity.CONTACT_ENV_VAR, "env@example.org")
    assert identity.crawl_contact() == "env@example.org"


def test_dotenv_file_is_read_when_env_is_unset(monkeypatch, tmp_path) -> None:
    (tmp_path / ".env").write_text(f"{identity.CONTACT_ENV_VAR}=dotenv@example.org\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv(identity.CONTACT_ENV_VAR, raising=False)
    assert identity.crawl_contact() == "dotenv@example.org"
