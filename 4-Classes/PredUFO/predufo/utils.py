
from typing import Optional, Iterable


def get_states_lbl(states: Optional[Iterable[str]]) -> str:
    """Create a label for a state or a set of states."""

    if states is None:
        states_lbl = 'Canada'

    elif len(list(states)) == 51:
        states_lbl = 'All States'
    else:
        states_lbl = '+'.join(states)

    return states_lbl
