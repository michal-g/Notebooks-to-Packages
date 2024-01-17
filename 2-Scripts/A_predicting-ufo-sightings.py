"""Applying autoregression to predict the # of UFOs that were seen in the US.

This script pulls down UFO sightings from the National UFO Reporting Center
website, and trains a time series regressor to predict the number of sightings
across the United States in the 1990s.

Example Usages
--------------
Really the only way to run this script is:
    python A_predicting-ufo-sightings.py

"""

import os
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


# we will be very careful to filter sightings that can be mapped to states
VALID_STATES = {
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
    'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
    }


def main():

    # Parsing and filtering data into formatted dataset #
    # ------------------------------------------------- #

    # create a table for the sightings data and only consider
    # valid unique sightings
    sights_df = pd.read_csv('../nuforc_events_complete.csv',
                            usecols=['event_time', 'city', 'state',
                                     'shape', 'duration', 'summary'])

    sights_df = sights_df.loc[sights_df.state.isin(VALID_STATES), :]

    # parse the date information into a more useful format
    sights_df['event_time'] = pd.to_datetime(
        sights_df.event_time, format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
    sights_df = sights_df.loc[~sights_df['event_time'].isna(), :]

    # Mapping state totals across entire time period #
    # ---------------------------------------------- #
    print("Producing plots of sightings by state...")

    # calculate totals across all time periods for each state and create a
    # local directory for saving plots
    state_totals = sights_df.groupby('state').size()
    os.makedirs(Path("map-plots", "gif-comps"), exist_ok=True)
    plt.rcParams["figure.figsize"] = (14, 9)

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])
    fig.update_layout(coloraxis_colorbar=dict(title="Sightings"))
    fig.write_image(Path("map-plots", "state-totals.png"), format='png')

    # Animating yearly state totals #
    # ----------------------------- #

    # create a Year x State table containing total weekly sightings for each
    # state; note that we have to take into account "missing" weeks that did
    # not have any sightings in any states
    state_yearlies = sights_df.groupby(
        [sights_df.event_time.dt.year, 'state']).size().unstack().fillna(0)

    # create a map of sightings by state for each week
    plt_files = list()
    for year, year_counts in state_yearlies.iterrows():
        state_locs = [str(x) for x in
                      year_counts.index.get_level_values('state')]

        fig = px.choropleth(locations=state_locs, locationmode="USA-states",
                            title=year, scope='usa',
                            color=year_counts.values, range_color=[0, 10],
                            color_continuous_scale=['white', 'red'])
        fig.update_layout(coloraxis_colorbar=dict(title="Sightings"))

        # save the map to file and keep track of the file name
        plt_file = Path("map-plots", "gif-comps", f"counts_{year}.png")
        fig.write_image(plt_file, format='png')
        plt_files += [imageio.v2.imread(plt_file)]

    # create an animation using the individual weekly maps
    imageio.mimsave(Path("map-plots", "counts.gif"), plt_files, duration=0.03)

    # Predicting weekly state totals #
    # ------------------------------ #

    pipeline = ForecasterPipeline([
        ('pre_scaler', StandardScaler()),
        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(num_lags=52)),
            ])),
        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

    print("Training a sightings prediction algorithm...")

    # assets and specially formatted objects used by the prediction pipeline
    # scikit-learn wants Xs to be 2-dimensional and ys to be 1-dimensional
    tscv = TimeSeriesSplit(n_splits=2)
    cali_yearlies = state_yearlies.CA.reindex(
        range(state_yearlies.index[0], state_yearlies.index[-1] + 1),
        fill_value=0.
        )

    cali_dates = cali_yearlies.index.values.reshape(-1, 1)
    cali_values = cali_yearlies.values

    real_values = list()
    pred_values = list()
    fig, ax = plt.subplots(figsize=(10, 6))

    # for each cross-validation fold, use the training samples in the fold to
    # train the pipeline and the remaining samples to test it
    for train_index, test_index in tscv.split(cali_yearlies):
        pipeline.fit(cali_dates[train_index], cali_values[train_index])
        preds = pipeline.predict(cali_dates[test_index], to_scale=True)

        # we'll keep track of the actual sightings and the predicted sightings
        # from each c-v fold for future reference
        real_values += cali_values[test_index].flatten().tolist()
        pred_values += preds.flatten().tolist()

        ax.plot(cali_dates[test_index], cali_values[test_index], color='black')
        ax.plot(cali_dates[test_index], preds, color='red')

    fig.savefig(Path("map-plots", "predictions.png"),
                bbox_inches='tight', format='png')

    # measure the quality of the predictions using root-mean-squared-error
    rmse_val = ((np.array(real_values)
                 - np.array(pred_values)) ** 2).mean() ** 0.5
    print(f"RMSE: {format(rmse_val, '.3f')}")


if __name__ == '__main__':
    main()
