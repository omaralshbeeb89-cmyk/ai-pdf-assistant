"""Password protection of the Streamlit app (no API calls, no real PDFs)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from pdf_assistant import config


APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def empty_project(tmp_path, monkeypatch):
    """Point the app at empty folders so it never loads real PDFs."""

    for name in ("PDF_DIR", "UPLOAD_DIR", "CACHE_DIR", "CONVERSATIONS_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())

    monkeypatch.setattr(config, "DEFAULT_PDF", tmp_path / "pdf_dir" / "none.pdf")


@pytest.fixture
def password(monkeypatch):
    monkeypatch.setattr(config, "get_app_password", lambda: "correct horse")
    return "correct horse"


@pytest.fixture
def no_password(monkeypatch):
    monkeypatch.setattr(config, "get_app_password", lambda: None)


def start():
    return AppTest.from_file(APP, default_timeout=30).run()


def sign_in(app, text):
    app.text_input[0].input(text)
    app.button[0].click()
    return app.run()


def is_app_open(app):
    return any("Upload a PDF" in info.value for info in app.info)


def test_without_password_the_app_opens(no_password):
    app = start()

    assert is_app_open(app)
    assert any("No password set" in caption.value for caption in app.sidebar.caption)


def test_with_password_only_the_login_form_is_shown(password):
    app = start()

    assert not is_app_open(app)
    assert len(app.text_input) == 1
    assert len(app.sidebar.button) == 0  # nothing behind the login is drawn


def test_wrong_password_is_refused(password):
    app = sign_in(start(), "guess")

    assert not is_app_open(app)
    assert "Wrong password. 4 attempt(s) left." in [error.value for error in app.error]


def test_right_password_opens_the_app_and_log_out_closes_it(password):
    app = sign_in(start(), password)

    assert is_app_open(app)

    next(button for button in app.sidebar.button if "Log out" in button.label).click()
    app.run()

    assert not is_app_open(app)
    assert len(app.text_input) == 1


def test_too_many_wrong_attempts_lock_the_form(password):
    app = start()

    for _ in range(5):
        app = sign_in(app, "guess")

    assert any("Too many wrong attempts" in error.value for error in app.error)

    app.run()  # still locked, even with the right password next

    assert len(app.text_input) == 0
    assert any("Try again in" in error.value for error in app.error)
