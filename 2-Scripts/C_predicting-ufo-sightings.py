"""Applying autoregression to predict the # of UFOs that were seen in the US.

This script pulls down UFO sightings from the National UFO Reporting Center
website, and trains a time series regressor to predict the number of sightings
in the United States across given ranges of years and states.

Example Usages
--------------
Predict total US sightings across the 90s:
    python C_predicting-ufo-sightings.py 1990 1999

Predict California sightings across the 80s:
    python C_predicting-ufo-sightings.py 1980 1989 --states CA

Predict New England ightings for five years with plots:
    python C_predicting-ufo-sightings.py 1987 1991 \
                --states ME MA VT NH RI --num-lags=2

Predict Oregonian sightings since 1950:
    python C_predicting-ufo-sightings.py 1950 2030 --states OR --num-lags=10

"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from skits.preprocessing import ReversibleImputer
from skits.pipeline import ForecasterPipeline
from skits.feature_extraction import AutoregressiveTransformer

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


def parse_sightings(start_year, end_year, states_lists, verbose):
    """Reading in raw data from UFO sightings website."""

    # create a table for the sightings data and only consider
    # valid unique sightings
    sights_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__),
                     'nuforc_events_complete.csv'),
        usecols=['event_time', 'city', 'state', 'shape', 'duration', 'summary']
        )

    states = {state for states_list in states_lists for state in states_list}
    sights_df = sights_df.loc[sights_df.state.isin(states), :]

    # parse the date information into a more useful format
    sights_df['event_time'] = pd.to_datetime(
        sights_df.event_time, format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
    sights_df = sights_df.loc[~sights_df['event_time'].isna(), :]

    state_yearlies = sights_df.groupby(
        [sights_df.event_time.dt.year, 'state']).size().unstack().fillna(0)
    state_yearlies.reindex(range(start_year, end_year + 1), fill_value=0.)

    if verbose > 1:
        print(f"Found {state_yearlies.values.sum()} unique sightings!")

    return state_yearlies


def plot_totals_map(yearly_sightings):
    """Mapping state totals across entire time period."""
    state_totals = yearly_sightings.sum()

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])
    fig.update_layout(coloraxis_colorbar=dict(title="Sightings"))

    fig.write_image(Path("map-plots", "state-totals.png"), format='png')


def animate_totals_map(yearly_sightings):
    """Plotting a gif of periodical state totals."""
    plt_files = list()

    # create a map of sightings by state for each time period
    for year, freq_counts in yearly_sightings.iterrows():
        state_locs = [str(x) for x in freq_counts.index]

        fig = px.choropleth(locations=state_locs,
                            locationmode="USA-states",
                            title=year, scope='usa',
                            color=freq_counts.values, range_color=[0, 10],
                            color_continuous_scale=['white', 'red'])
        fig.update_layout(coloraxis_colorbar=dict(title="Sightings"))

        plt_file = Path("map-plots", "gif-comps", f"counts_{year}.png")
        fig.write_image(plt_file, format='png')
        plt_files += [imageio.v2.imread(plt_file)]

    # create an animation using the individual weekly maps
    imageio.mimsave(Path("map-plots", "counts.gif"), plt_files, duration=0.03)


def predict_sightings(yearly_sightings, states, num_lags,
                      create_plots=False, verbose=0):
    """Predicting weekly state totals."""

    pipeline = ForecasterPipeline([
        ('pre_scaler', StandardScaler()),
        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(num_lags=num_lags))])),
        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

    # assets and specially formatted objects used by the prediction pipeline
    # scikit-learn wants Xs to be 2-dimensional and ys to be 1-dimensional
    tscv = TimeSeriesSplit(n_splits=2)
    pred_values = yearly_sightings.sum(axis=1)
    pred_dates = yearly_sightings.index.values.reshape(-1, 1)

    if verbose > 1:
        print(f"There are {pred_values.sum()} total sightings for "
              f"{get_states_lbl(states)}, of which the maximum "
              f"({pred_values.max()}) took place on "
              f"{pred_values.idxmax().strftime('%F')}!")

    date_values = list()
    real_values = list()
    regr_values = list()

    # for each cross-validation fold, use the training samples in the fold to
    # train the pipeline and the remaining samples to test it
    for train_index, test_index in tscv.split(pred_values.values):
        try:
            pipeline.fit(pred_dates[train_index],
                         pred_values.values[train_index])
            preds = pipeline.predict(pred_dates[test_index],
                                     to_scale=True)

        except ValueError:
            preds = np.array([0] * len(test_index))

        # we'll keep track of the actual sightings and the predicted sightings
        # from each c-v fold for future reference
        date_values += [pred_dates[test_index]]
        real_values += [pred_values.values[test_index].flatten().tolist()]
        regr_values += [preds.flatten().tolist()]

    if create_plots:
        plot_predictions(date_values, real_values, regr_values)

    # measure the quality of the predictions using root-mean-squared-error
    rmse_val = ((np.array(real_values)
                 - np.array(regr_values)) ** 2).mean() ** 0.5

    return regr_values, rmse_val


def plot_predictions(date_values, real_values, regr_values):
    """Utility function for plotting predicted sightings vs. actual counts."""
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

    parser.add_argument("--num-lags",
                        type=int, default=5, dest="num_lags",
                        help="the number of time series features to use in "
                             "auto-regression")
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

    # create a Year x State table containing total periodical sightings
    # for each state; note that we have to take into account "missing" periods
    # that did not have any sightings in any states
    state_yearlies = parse_sightings(*args.years, states_lists, args.verbose)

    if args.create_plots:
        os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
        plot_totals_map(state_yearlies)
        animate_totals_map(state_yearlies)

    # predict sightings for each given set of states
    for pred_states in states_lists:
        pred_sightings, rmse_val = predict_sightings(
            yearly_sightings=state_yearlies, states=pred_states,
            num_lags=args.num_lags, create_plots=args.create_plots,
            verbose=args.verbose,
            )

        print(f"{get_states_lbl(pred_states)}"
              f"\tRMSE: {format(rmse_val, '.3f')}")


if __name__ == '__main__':
    main()
