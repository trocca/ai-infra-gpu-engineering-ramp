"""Shared test plumbing. COMPLETE — do not edit.

`call` runs skeleton code and reports SKIPPED (not implemented yet) instead of
ERROR while a function still raises NotImplementedError.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


@pytest.fixture(name="call")
def call_fixture():
    return call
