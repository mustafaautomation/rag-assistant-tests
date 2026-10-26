from __future__ import annotations

import pytest

from rag_assistant.assistant import Assistant


@pytest.fixture(scope="session")
def assistant() -> Assistant:
    return Assistant.default()
