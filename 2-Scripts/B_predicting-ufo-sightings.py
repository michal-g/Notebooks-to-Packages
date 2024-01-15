"""Applying autoregression to predict the # of UFOs that were seen in the US.

This script pulls down UFO sightings from the National UFO Reporting Center
website, and trains a time series regressor to predict the number of sightings
in the United States across a given range of years and states.

See
https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
for the list of valid frequency aliases to use for `window`.

Example Usages
--------------
Predict total US weekly sightings across the 90s:
    python B_predicting-ufo-sightings.py 1990 1999

Predict California weekly sightings across the 80s:
    python B_predicting-ufo-sightings.py 1980 1989 --states CA

Predict New England monthly sightings and make plots:
    python B_predicting-ufo-sightings.py 1977 1991 \
                --states ME MA VT NH RI --window M \
                --num-lags=12 --seasonal-period=12 -p

Predict biweekly Oregonian sightings since 1950:
    python B_predicting-ufo-sightings.py 1950 2030 --states OR --window 2W

"""

import os
import argparse
import itertools
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from skits.preprocessing import ReversibleImputer
from skits.pipeline import ForecasterPipeline
from skits.feature_extraction import (AutoregressiveTransformer,
                                      SeasonalTransformer)

import imageio
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14, 9)


# we will be very careful to filter sightings that can be mapped to states
VALID_STATES = {
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
    'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
    }


def main():
    # Grabbing parameter values from the command line #
    # ----------------------------------------------- #
    parser = argparse.ArgumentParser()

    parser.add_argument("years",
                        type=int, nargs=2,
                        help="the range of years (inclusive) whose sightings "
                             "will be considered")

    parser.add_argument("--states",
                        type=str, nargs='+',
                        default=VALID_STATES, choices=VALID_STATES,
                        help="which states' sightings to predict "
                             "— the default is to use all states")

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

    # Parsing and filtering data into formatted dataset #
    # ------------------------------------------------- #

    # create a table for the sightings data and only consider
    # valid unique sightings
    sights_df = pd.read_csv('../nuforc_events_complete.csv',
                            usecols=['event_time', 'city', 'state',
                                     'shape', 'duration', 'summary'])

    sights_df = sights_df.loc[sights_df.state.isin(VALID_STATES), :]

    # parse the date information into a more useful format
    sights_df['event_time'] = pd.to_datetime(sights_df.event_time,
                                             format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
    sights_df = sights_df.loc[~sights_df['event_time'].isna(), :]

    if args.verbose > 1:
        print(f"Found {sights_df.shape[0]} unique sightings!")

    # Mapping state totals across entire time period #
    # ---------------------------------------------- #

    if args.verbose:
        print("Producing plots of sightings by state...")

    # calculate totals across all time periods for each state and create a
    # local directory for saving plots
    if args.create_plots:
        state_totals = sights_df.groupby('State').size()
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)

        fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                            scope="usa", locationmode="USA-states",
                            color=state_totals.values,
                            range_color=[0, state_totals.max()],
                            color_continuous_scale=['white', 'red'])
        fig.write_image(Path("map-plots", "state-totals.png"), format='png')

    # Animating periodical state totals #
    # --------------------------------- #

    # create a Time Period x State table containing total periodical sightings
    # for each state; note that we have to take into account "missing" periods
    # that did not have any sightings in any states
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
        plt_files = list()

        # create a map of sightings by state for each week
        for week, week_counts in state_byfreq.iterrows():
            day_lbl = week.strftime('%F')
            state_locs = [str(x) for x in
                          week_counts.index.get_level_values('State')]

            fig = px.choropleth(locations=state_locs,
                                locationmode="USA-states",
                                title=day_lbl, scope='usa',
                                color=week_counts.values, range_color=[0, 10],
                                color_continuous_scale=['white', 'black'])

            plt_file = Path("map-plots", "gif-comps", f"counts_{day_lbl}.png")
            fig.write_image(plt_file, format='png')
            plt_files += [imageio.v2.imread(plt_file)]

        # create an animation using the individual weekly maps
        imageio.mimsave(Path("map-plots", "counts.gif"), plt_files,
                        duration=0.03)

    # Predicting periodical state totals #
    # ---------------------------------- #

    pipeline = ForecasterPipeline([
        ('pre_scaler', StandardScaler()),

        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(
                num_lags=args.num_lags)),
            ('seasonal_features', SeasonalTransformer(
                seasonal_period=args.seasonal_period)),
            ])),

        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

    if args.verbose:
        print("Training a sightings prediction algorithm...")

    # assets and specially formatted objects used by the prediction pipeline
    # scikit-learn wants Xs to be 2-dimensional and ys to be 1-dimensional
    tscv = TimeSeriesSplit(n_splits=4)
    pred_byfreq = state_byfreq.loc[:, list(args.states)].sum(axis=1)
    pred_dates = pred_byfreq.index.values.reshape(-1, 1)
    pred_values = pred_byfreq.values

    if args.verbose > 2:
        if len(args.states) == 51:
            states_lbl = 'All States'
        else:
            states_lbl = '+'.join(args.states)

        print(f"There are {pred_byfreq.sum()} total sightings for "
              f"{states_lbl}, of which the maximum ({pred_byfreq.max()}) took "
              f"place on {pred_byfreq.idxmax().strftime('%F')}!")

    real_values = list()
    regr_values = list()

    if args.create_plots:
        fig, ax = plt.subplots(figsize=(10, 6))

    # for each cross-validation fold, use the training samples in the fold to
    # train the pipeline and the remaining samples to test it
    for train_index, test_index in tscv.split(pred_byfreq):
        pipeline.fit(pred_dates[train_index], pred_values[train_index])
        preds = pipeline.predict(pred_dates[test_index], to_scale=True)

        # we'll keep track of the actual sightings and the predicted sightings
        # from each c-v fold for future reference
        real_values += pred_values[test_index].flatten().tolist()
        regr_values += preds.flatten().tolist()

        if args.create_plots:
            ax.plot(pred_dates[test_index], pred_values[test_index],
                    color='black')
            ax.plot(pred_dates[test_index], preds, color='red')

    if args.create_plots:
        fig.savefig(Path("map-plots", "predictions.png"),
                    bbox_inches='tight', format='png')

    # measure the quality of the predictions using root-mean-squared-error
    rmse_val = ((np.array(real_values)
                 - np.array(regr_values)) ** 2).mean() ** 0.5
    print(f"RMSE: {format(rmse_val, '.3f')}")


if __name__ == '__main__':
    main()
