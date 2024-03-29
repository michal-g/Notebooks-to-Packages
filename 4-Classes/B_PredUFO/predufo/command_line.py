"""Command line interfaces for predicting sightings in different regions."""

import os
import argparse
from pathlib import Path
from typing import Optional, Iterable
from .data import AmericanSightingsDataset, CanadianSightingsDataset


def get_states_lbl(states: Optional[Iterable[str]]) -> str:
    """Create a label for a state or a set of states."""

    if states is None:
        states_lbl = 'Canada'

    elif len(list(states)) == 51:
        states_lbl = 'All States'
    else:
        states_lbl = '+'.join(states)

    return states_lbl


# Base parser class for interfaces predicting sightings #
# ----------------------------------------------------- #
parent_parser = argparse.ArgumentParser(add_help=False)

parent_parser.add_argument("years",
                           type=int, nargs=2,
                           help="the range of years (inclusive) whose "
                                "sightings will be considered")

parent_parser.add_argument("--num-lags",
                           type=int, default=5, dest="num_lags",
                           help="the number of time series features to use in "
                                "auto-regression")

parent_parser.add_argument("--create-plots", "-p",
                           action='store_true', dest="create_plots",
                           help="save visualizations to file?")
parent_parser.add_argument("--verbose", "-v", action='count', default=0)


def predict_usa():
    """Source code for the predUFO-USA command line interface."""
    parser = argparse.ArgumentParser(parents=[parent_parser])

    parser.add_argument("--states",
                        type=str, nargs='+', action='append',
                        choices=AmericanSightingsDataset.VALID_STATES,
                        help="which states' sightings to predict — the "
                             "default is to use all states, can be repeated "
                             "for predicting sightings for different "
                             "sets of states")

    args = parser.parse_args()

    # allowing for multiple sets of states means we have to handle the special
    # case where no sets were defined manually — argparse is buggy otherwise
    if args.states:
        states_lists = sorted(args.states)
    else:
        states_lists = [list(AmericanSightingsDataset.VALID_STATES)]

    sights_data = AmericanSightingsDataset(*args.years, verbosity=args.verbose)

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        sights_data.plot_totals_map()
        sights_data.animate_totals_map()

    # predict sightings for each given set of states
    for pred_states in states_lists:
        states_by_freq = sights_data.get_state_sightings(
            args.window, pred_states)

        pred_sightings, rmse_val = states_by_freq.predict(
            num_lags=args.num_lags, seasonal_period=args.seasonal_period,
            create_plots=args.create_plots, verbose=args.verbose,
            )

        print(f"{get_states_lbl(pred_states)}"
              f"\tRMSE: {format(rmse_val, '.3f')}")


def predict_canada():
    """Source code for the predUFO-Canada command line interface."""
    parser = argparse.ArgumentParser(parents=[parent_parser])
    args = parser.parse_args()

    # create a table containing sightings by time period; note that we have to
    # take into account "missing" periods that did not have any sightings
    sights_data = CanadianSightingsDataset(*args.years, verbosity=args.verbose)
    canada_byfreq = sights_data.get_sightings(args.window)

    # predict sightings for all of Canada
    pred_sightings, rmse_val = canada_byfreq.predict(
        num_lags=args.num_lags, seasonal_period=args.seasonal_period,
        create_plots=args.create_plots, verbose=args.verbose,
        )

    print(f"{get_states_lbl(None)}\tRMSE: {format(rmse_val, '.3f')}")
