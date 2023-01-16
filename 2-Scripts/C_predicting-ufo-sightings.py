"""Applying autoregression to predict the # of UFOs that were seen in the US.

Example Usages
--------------
Predicting total US weekly sightings across the 90s:
    python C_predicting-ufo-sightings.py 1990 1999

Predicting California weekly sightings across the 80s:
    python C_predicting-ufo-sightings.py 1980 1989 --states CA

Predicting New England monthly sightings for five years with plots:
    python C_predicting-ufo-sightings.py 1987 1991 \
                --states ME MA VT NH RI --window M \
                --num-lags=12 --seasonal-period=12

Predicting biweekly Oregonian sightings since 1950:
    python C_predicting-ufo-sightings.py 1950 2030 --states OR --window 2W

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


def get_states_lbl(states):
    """Create a label for a state or a set of states."""

    if len(states) == 51:
        states_lbl = 'All States'
    else:
        states_lbl = '+'.join(states)

    return states_lbl


def scrape_sightings(first_year, last_year, verbose):
    """Reading in raw data from UFO sightings website."""

    # initialize assets for scraping the reports portal
    base_url = 'https://nuforc.org/webreports'
    grab = requests.get('/'.join([base_url, 'ndxevent.html']))

    # initialize data structures for storing parsed data
    sightings = []
    col_labels = ['Date', 'City', 'State', 'Country', 'Shape', 'Duration',
                  'Summary', 'Posted', 'Images']

    # create a regular expression matching our range of years
    year_regex = "({})".format(
        '|'.join([str(year)
                  for year in range(first_year, last_year + 1)])
        )

    # for each link to a month's data, create assets for scraping that table
    for month_link in BeautifulSoup(grab.text, 'html.parser')(
            'a', string=re.compile(f"[0-9]{{2}}\/{year_regex}")):
        month_grab = requests.get('/'.join([base_url, month_link.get('href')]))

        # the HTML formatting is kind of weird; we first grab the outermost of
        # a recursively defined set of table elements
        table_data = BeautifulSoup(
            month_grab.text, 'html.parser')('tr')[1]('td')
        cur_sighting = None

        # then we loop over a set of table entries that are defined as one big
        # row??? maybe there's an easier way to do this but that's ok
        for lbl, col in zip(itertools.cycle(col_labels), table_data):
            if lbl == 'Date':
                if cur_sighting is not None:
                    sightings.append(cur_sighting)

                cur_sighting = {'Date': col.string}

            # start a new sighting record, after adding the last record to the
            # list of sightings if this is not the first row
            else:
                cur_sighting[lbl] = col.string

        # accounting for the last row
        if cur_sighting is not None:
            sightings.append(cur_sighting)

    # create a table for the sightings data and only consider
    # valid unique sightings
    sights_df = pd.DataFrame(sightings).drop_duplicates()
    sights_df = sights_df.loc[(sights_df.Country == 'USA')
                              & sights_df.State.isin(VALID_STATES), :]

    # parse the date information into more useful format
    sights_df['Date'] = pd.to_datetime(
        [dt.split()[0] for dt in sights_df['Date']], format='%m/%d/%y')

    if verbose:
        print(f"Found {sights_df.shape[0]} unique sightings!")

    return sights_df


def plot_totals_map(sights_df):
    """Mapping state totals across entire time period."""
    state_totals = sights_df.groupby('State').size()

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])

    fig.write_image(Path("map-plots", "state-totals.png"), format='png')


def animate_totals_map(weeklies):
    plt_files = list()

    # create a map of sightings by state for each week
    for week, week_counts in weeklies.iterrows():
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


def predict_sightings(weeklies, states, num_lags, seasonal_period,
                      create_plots=False, verbose=0):
    """Predicting weekly state totals."""

    pipeline = ForecasterPipeline([
        ('pre_scaler', StandardScaler()),

        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(
                num_lags=num_lags)),
            ('seasonal_features', SeasonalTransformer(
                seasonal_period=seasonal_period)),
            ])),

        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

    # assets and specially formatted objects used by the prediction pipeline
    tscv = TimeSeriesSplit(n_splits=4)
    pred_byfreq = weeklies.loc[:, list(states)].sum(axis=1)
    pred_dates = pred_byfreq.index.values.reshape(-1, 1)
    pred_values = pred_byfreq.values

    if verbose > 1:
        print(f"There are {pred_byfreq.sum()} total sightings for "
              f"{get_states_lbl(states)}, of which the maximum "
              f"({pred_byfreq.max()}) took place on "
              f"{pred_byfreq.idxmax().strftime('%F')}!")

    date_values = list()
    real_values = list()
    regr_values = list()

    # for each cross-validation fold, use the training samples in the fold to
    # train the pipeline and the remaining samples to test it
    for train_index, test_index in tscv.split(pred_byfreq):
        pipeline.fit(pred_dates[train_index], pred_values[train_index])
        preds = pipeline.predict(pred_dates[test_index], to_scale=True)

        # we'll keep track of the actual sightings and the predicted sightings
        # from each c-v fold for future reference
        date_values += [pred_dates[test_index]]
        real_values += [pred_values[test_index].flatten().tolist()]
        regr_values += [preds.flatten().tolist()]

    if create_plots:
        plot_predictions(date_values, real_values, regr_values)

    # measure the quality of the predictions using root-mean-squared-error
    rmse_val = ((np.array(real_values)
                 - np.array(regr_values)) ** 2).mean() ** 0.5

    return regr_values, rmse_val


def plot_predictions(date_values, real_values, regr_values):
    fig, ax = plt.subplots(figsize=(10, 6))

    for dates, reals, regrs in zip(date_values, real_values, regr_values):
        ax.plot(dates, reals, color='black')
        ax.plot(dates, regrs, color='red')

    fig.savefig(Path("map-plots", "predictions.png"),
                bbox_inches='tight', format='png')


def main():
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


if __name__ == '__main__':
    main()
