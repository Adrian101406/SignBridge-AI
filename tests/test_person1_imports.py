from __future__ import annotations

import pytest

from signbridge_app.config import DEFAULT_SETTINGS
from signbridge_app.recognition_service import Person1RecognizerLoader


@pytest.mark.parametrize("domain", ("number", "medical", "general"))
def test_canonical_person1_recognizer_loads(domain: str) -> None:
    loader = Person1RecognizerLoader(DEFAULT_SETTINGS)

    recognizer = loader.get(domain)

    assert recognizer is not None
    loader.close()
