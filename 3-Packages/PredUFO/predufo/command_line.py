"""Command line interfaces for predicting sightings in different regions."""

import os
import argparse
from pathlib import Path
import pandas as pd

from .data import VALID_STATES, scrape_sightings
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

parent_parser.add_argument("--window",
                           type=str, default='W',
                           help="which time period to use to group sightings "
                                "— the default is weekly, and any other "
                                "pandas offset alias can be used")

parent_parser.add_argument("--num-lags",
                           type=int, default=52, dest="num_lags",
                           help="the number of time series features to use in "
                                "auto-regression")
parent_parser.add_argument("--seasonal-period",
                           type=int, default=52, dest="seasonal_period",
                           help="the number of time points in a season to use "
                                "for seasonal correction")

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
        print("Reading in American sightings from HTML inputs...")

    # create a Time Period x State table containing total periodical sightings
    # for each state; note that we have to take into account "missing" periods
    # that did not have any sightings in any states
    sights_df = scrape_sightings(*args.years,
                                 country='usa', verbose=args.verbose)

    state_table = sights_df.groupby(
        ['Date', 'State']).size().unstack().fillna(0)

    state_table = state_table.reindex(
        index=pd.date_range(f"01-01-{args.years[0]}",
                            f"12-31-{args.years[1]}"),
        fill_value=0
        ).sort_index()

    state_byfreq = state_table.groupby(
        pd.Grouper(axis=0, freq=args.window, sort=True)).sum().astype(int)

    if args.verbose:
        print("Producing plots of sightings by state...")

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        plot_totals_map(sights_df)
        animate_totals_map(state_byfreq)

    if args.verbose:
        print("Training American sightings prediction algorithms...")

    # predict sightings for each given set of states
    for pred_states in states_lists:
        pred_sightings, rmse_val = predict_sightings(
            sightings=state_byfreq, num_lags=args.num_lags,
            seasonal_period=args.seasonal_period, states=pred_states,
            create_plots=args.create_plots, verbose=args.verbose,
            )

        print(f"{get_states_lbl(pred_states)}"
              f"\tRMSE: {format(rmse_val, '.3f')}")


def predict_canada():
    """Source code for the predUFO-Canada command line interface."""
    parser = argparse.ArgumentParser(parents=[parent_parser])
    args = parser.parse_args()

    if args.verbose:
        print("Reading in Canadian sightings from HTML inputs...")

    # create a table containing sightings by time period; note that we have to
    # take into account "missing" periods that did not have any sightings
    sights_df = scrape_sightings(*args.years,
                                 country='canada', verbose=args.verbose)

    date_table = sights_df.groupby('Date').size().reindex(
        index=pd.date_range(f"01-01-{args.years[0]}",
                            f"12-31-{args.years[1]}"),
        fill_value=0
        ).sort_index()

    canada_byfreq = date_table.groupby(
        pd.Grouper(freq=args.window)).sum().astype(int)

    if args.verbose:
        print("Training a Canadian sightings prediction algorithm...")

    # predict sightings for all of Canada
    pred_sightings, rmse_val = predict_sightings(
        sightings=canada_byfreq, num_lags=args.num_lags,
        seasonal_period=args.seasonal_period, states=None,
        create_plots=args.create_plots, verbose=args.verbose,
        )

    print(f"{get_states_lbl(None)}\tRMSE: {format(rmse_val, '.3f')}")
