
def get_states_lbl(states: list[str]) -> str:
    """Create a label for a state or a set of states."""

    if len(states) == 51:
        states_lbl = 'All States'
    else:
        states_lbl = '+'.join(states)

    return states_lbl
