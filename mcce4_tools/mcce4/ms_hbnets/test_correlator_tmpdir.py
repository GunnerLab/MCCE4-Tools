#!/usr/bin/env python3

"""
Regression tests for GunnerLab/MCCE4-Tools issue #200.

`ms_hbnets states` crashed with:

    TypeError: TemporaryDirectory.__init__() got an unexpected keyword
    argument 'delete'

because the correlation pipeline constructed its working folder with
`TemporaryDirectory(delete=False)`. The `delete` keyword only exists on
Python 3.12+, while this project pins `requires-python = "==3.10.*"`
(pyproject.toml). On the pinned interpreter the keyword raises at
construction time, before any matrix work starts.

The fix drops the `delete` kwarg at both call sites
(HbCorrelator.run_pipeline_pairs_corr and HbCorrelator.run_pipeline_da_corr).
That is a no-op behavior change: each site already calls `tmp_dir.cleanup()`
explicitly on every exit path (including the early `if not self.ok` return),
and `TemporaryDirectory.cleanup()` is idempotent.
"""

import inspect
import sys
from tempfile import TemporaryDirectory

from mcce4.ms_hbnets import correlator


def _environment_reproduces_the_issue():
    """True when the running interpreter rejects the `delete` kwarg, i.e.
    exactly the interpreter class (3.10) where issue #200 crashes. On
    3.12+ the kwarg exists, so the crash could not occur there either way."""
    if sys.version_info >= (3, 12):
        return False
    try:
        TemporaryDirectory(delete=False)
        return False
    except TypeError:
        return True


def test_pinned_interpreter_accepts_the_pipeline_tmp_dir_construction():
    """Constructing TemporaryDirectory() the way the pipeline now does must
    not raise on the pinned interpreter (3.10)."""
    tmp_dir = TemporaryDirectory()
    try:
        assert tmp_dir.name
    finally:
        tmp_dir.cleanup()


def test_pairs_corr_pipeline_uses_the_portable_tmp_dir_form():
    """run_pipeline_pairs_corr must build its temp dir without the
    3.12-only `delete` kwarg. The source check is the regression guard:
    it fails pre-fix (source contains the kwarg) and passes post-fix."""
    src = inspect.getsource(correlator.HbCorrelator.run_pipeline_pairs_corr)
    assert "delete=" not in src, (
        "run_pipeline_pairs_corr passes the 3.12-only `delete` kwarg to "
        "TemporaryDirectory; the pinned interpreter (3.10) rejects it "
        "(issue #200). Both sites already clean up explicitly via "
        ".cleanup() (idempotent), so the kwarg is not needed."
    )
    # Where the issue actually reproduces, prove the kwarg form really
    # raises, so the source guard above is tied to a live failure.
    if _environment_reproduces_the_issue():
        try:
            TemporaryDirectory(delete=False)
        except TypeError:
            pass  # expected on 3.10: the kwarg form is what crashed the CLI
        else:
            raise AssertionError(
                "Environment does not reproduce issue #200: the `delete` "
                "kwarg is accepted on %s" % sys.version.split()[0]
            )


def test_da_corr_pipeline_uses_the_portable_tmp_dir_form():
    """Same guard for the donor/acceptor pipeline (sibling call site)."""
    src = inspect.getsource(correlator.HbCorrelator.run_pipeline_da_corr)
    assert "delete=" not in src, (
        "run_pipeline_da_corr passes the 3.12-only `delete` kwarg to "
        "TemporaryDirectory; the pinned interpreter (3.10) rejects it "
        "(issue #200). Both sites already clean up explicitly via "
        ".cleanup() (idempotent), so the kwarg is not needed."
    )
    if _environment_reproduces_the_issue():
        try:
            TemporaryDirectory(delete=False)
        except TypeError:
            pass  # expected on 3.10
        else:
            raise AssertionError(
                "Environment does not reproduce issue #200: the `delete` "
                "kwarg is accepted on %s" % sys.version.split()[0]
            )
