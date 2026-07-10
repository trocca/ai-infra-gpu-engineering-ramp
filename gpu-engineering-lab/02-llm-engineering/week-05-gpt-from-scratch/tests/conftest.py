"""Shared test plumbing. COMPLETE — do not edit.

Tests call skeleton code through the `call` helper: while a function still
raises NotImplementedError the test is reported as SKIPPED (not implemented
yet) instead of ERROR, so `make test` is runnable from day 1 and turns green
as you implement.
"""

import sys
from pathlib import Path

import pytest

# Make `import src.*` work no matter where pytest is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def call(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"not implemented yet: {e}")


@pytest.fixture(name="call")
def call_fixture():
    return call
