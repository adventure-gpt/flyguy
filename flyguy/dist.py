"""Several GPUs: one trainer process per card, every optimiser step on the mean gradient of all of them.

The harness that starts those processes is not part of this repository; here there is one process, so nothing is
shared. flyguy/fly_native.py asks on() before every step of the fly actor's optimiser."""
from __future__ import annotations


def on() -> bool:
    return False


def mean_(tensors) -> None:
    """In place: every tensor becomes the mean of that tensor over all trainer processes."""
