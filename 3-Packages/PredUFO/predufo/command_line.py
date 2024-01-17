"""Command line interfaces for predicting sightings in different regions."""

import os
import argparse
from pathlib import Path

from .data import VALID_STATES, VALID_PROVINCES, parse_sightings
from .plot import plot_totals_map, animate_totals_map
from .predict import predict_sightings
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
                        action='append', choices=VALID_STATES,
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
        states_lists = [list(VALID_STATES)]

    if args.verbose:
        print("Reading in American sightings from file...")

    # create a Year x State table containing total periodical sightings
    # for each state; note that we have to take into account "missing" periods
    # that did not have any sightings in any states
    yearly_sightings = parse_sightings(*args.years, country='usa',
                                       verbose=args.verbose)

    if args.verbose:
        print("Producing plots of sightings by state...")

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        plot_totals_map(yearly_sightings)
        animate_totals_map(yearly_sightings)

    if args.verbose:
        print("Training American sightings prediction algorithms...")

    # predict sightings for each given set of states
    for pred_states in states_lists:
        pred_sightings, rmse_val = predict_sightings(
            sightings=yearly_sightings, num_lags=args.num_lags,
            states=pred_states, create_plots=args.create_plots,
            verbose=args.verbose,
            )

        print(f"{get_states_lbl(pred_states)}"
              f"\tRMSE: {format(rmse_val, '.3f')}")


def predict_canada():
    """Source code for the predUFO-Canada command line interface."""
    parser = argparse.ArgumentParser(parents=[parent_parser])
    args = parser.parse_args()

    if args.verbose:
        print("Reading in Canadian sightings from file...")

    # create a table containing sightings by time period; note that we have to
    # take into account "missing" periods that did not have any sightings
    yearly_sightings = parse_sightings(*args.years, country='canada',
                                       verbose=args.verbose)

    if args.verbose:
        print("Training a Canadian sightings prediction algorithm...")

    if args.create_plots:
        os.makedirs("map-plots", exist_ok=True)

    # predict sightings for all of Canada
    pred_sightings, rmse_val = predict_sightings(
        sightings=yearly_sightings, num_lags=args.num_lags, states=None,
        create_plots=args.create_plots, verbose=args.verbose,
        )

    print(f"{get_states_lbl(VALID_PROVINCES)}"
          f"\tRMSE: {format(rmse_val, '.3f')}")
