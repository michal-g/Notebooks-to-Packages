
import os
import argparse
from pathlib import Path
import pandas as pd

from .data import VALID_STATES, scrape_sightings
from .plot import plot_totals_map, animate_totals_map
from .predict import predict_sightings
from .utils import get_states_lbl


def predict_usa():
    parser = argparse.ArgumentParser()

    parser.add_argument("years",
                        type=int, nargs=2,
                        help="the range of years (inclusive) whose sightings "
                             "will be considered")

    parser.add_argument("--states",
                        type=str, nargs='+',
                        action='append', choices=VALID_STATES,
                        help="which states' sightings to predict — the "
                             "default is to use all states, can be repeated "
                             "for predicting sightings for different "
                             "sets of states")

    parser.add_argument("--window",
                        type=str, default='W',
                        help="which time period to use to group sightings — "
                             "the default is weekly, and any other pandas "
                             "offset alias can be used")

    parser.add_argument("--num-lags",
                        type=int, default=52, dest="num_lags",
                        help="the number of time series features to use in "
                             "auto-regression")
    parser.add_argument("--seasonal-period",
                        type=int, default=52, dest="seasonal_period",
                        help="the number of time points in a season to use "
                             "for seasonal correction")

    parser.add_argument("--create-plots", "-p",
                        action='store_true', dest="create_plots",
                        help="save visualizations to file?")
    parser.add_argument("--verbose", "-v", action='count', default=0)

    args = parser.parse_args()

    # allowing for multiple sets of states means we have to handle the special
    # case where no sets were defined manually — argparse is buggy otherwise
    if args.states:
        states_lists = sorted(args.states)
    else:
        states_lists = [list(VALID_STATES)]

    # create a Week x State table containing total weekly sightings for each
    # state; note that we have to take into account "missing" weeks that did
    # not have any sightings in any states
    sights_df = scrape_sightings(*args.years, args.verbose)

    state_table = sights_df.groupby(
        ['Date', 'State']).size().unstack().fillna(0)

    state_table = state_table.reindex(
        index=pd.date_range(f"01-01-{args.years[0]}",
                            f"12-31-{args.years[1]}"),
        fill_value=0
        ).sort_index()

    state_byfreq = state_table.groupby(
        pd.Grouper(axis=0, freq=args.window, sort=True)).sum().astype(int)

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        plot_totals_map(sights_df)
        animate_totals_map(state_byfreq)

    for pred_states in states_lists:
        pred_sightings, rmse_val = predict_sightings(
            state_byfreq, pred_states, args.num_lags, args.seasonal_period,
            args.create_plots, args.verbose,
            )

        print(f"{get_states_lbl(pred_states)}"
              f"\tRMSE: {format(rmse_val, '.3f')}")
