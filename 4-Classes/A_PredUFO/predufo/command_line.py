"""Command line interfaces for predicting sightings in different regions."""

import os
import argparse
from pathlib import Path
from .sightings import SightingsDataset
from .utils import get_states_lbl


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
                        type=str, nargs='+',
                        action='append', choices=SightingsDataset.VALID_STATES,
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
        states_lists = [list(SightingsDataset.VALID_STATES)]

    sights_data = SightingsDataset(country='usa',
                                   verbosity=args.verbose)

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        sights_data.plot_totals_map()

    for pred_states in states_lists:
        state_byfreq = sights_data.get_sightings(*args.years, pred_states)

        if args.create_plots:
            state_byfreq.animate_totals_map()

        # predict sightings for each given set of states
        pred_sightings, rmse_val = state_byfreq.predict(
            num_lags=args.num_lags, states=pred_states,
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
    sights_data = SightingsDataset(country='canada',
                                   verbosity=args.verbose)
    canada_byfreq = sights_data.get_sightings(*args.years)

    # predict sightings for all of Canada
    pred_sightings, rmse_val = canada_byfreq.predict(
        num_lags=args.num_lags, states=None,
        create_plots=args.create_plots, verbose=args.verbose,
        )

    print(f"{get_states_lbl(None)}\tRMSE: {format(rmse_val, '.3f')}")
