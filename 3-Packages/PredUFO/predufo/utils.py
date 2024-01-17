
from .data import VALID_STATES, VALID_PROVINCES
from typing import Optional, Iterable


def get_states_lbl(states: Optional[Iterable[str]]) -> str:
    """Create a label for a state or a set of states."""

    if set(states) == VALID_STATES:
        states_lbl = 'USA'
    elif set(states) == VALID_PROVINCES:
        states_lbl = 'Canada'

    else:
        states_lbl = '+'.join(states)

    return states_lbl
