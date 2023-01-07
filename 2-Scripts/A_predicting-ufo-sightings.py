"""Applying autoregression to predict the # of UFOs that were seen in the US.

Example Usages
--------------
Really the only way to run this script is:
    python A_predicting-ufo-sightings.py

"""

import os
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
    # Reading in raw data from UFO sightings website #
    # ---------------------------------------------- #

    # initialize assets for scraping the reports portal
    base_url = 'https://nuforc.org/webreports'
    grab = requests.get('/'.join([base_url, 'ndxevent.html']))

    # initialize data structures for storing parsed data
    sightings = []
    col_labels = ['Date', 'City', 'State', 'Country', 'Shape', 'Duration',
                  'Summary', 'Posted', 'Images']

    # for each link to a month's data, create assets for scraping that table
    for month_link in BeautifulSoup(grab.text, 'html.parser')(
            'a', string=re.compile("[0-9]{2}\/199[0-9]")):
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

    # Parsing and filtering data into formatted dataset #
    # ------------------------------------------------- #

    # create a table for the sightings data and only consider
    # valid unique sightings
    sights_df = pd.DataFrame(sightings).drop_duplicates()
    sights_df = sights_df.loc[(sights_df.Country == 'USA')
                              & sights_df.State.isin(VALID_STATES), :]

    # parse the date information into more useful format
    sights_df['Date'] = pd.to_datetime(
        [dt.split()[0] for dt in sights_df['Date']],
        format='%m/%d/%y')

    # Mapping state totals across entire time period #
    # ---------------------------------------------- #

    state_totals = sights_df.groupby('State').size()
    os.makedirs("map-plots", exist_ok=True)

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])

    # Animating weekly state totals #
    # ----------------------------- #

    # create a Week x State table containing total weekly sightings for each
    # state; note that we have to take into account "missing" weeks that did
    # not have any sightings in any states
    state_table = sights_df.groupby(
        ['Date', 'State']).size().unstack().fillna(0)
    state_table = state_table.reindex(index=pd.date_range('01-01-1990',
                                                          '12-31-1999'),
                                      fill_value=0).sort_index()

    state_weeklies = state_table.groupby(
        pd.Grouper(axis=0, freq='W', sort=True)).sum()

    # create a map of sightings by state for each week
    plt_files = list()
    for week, week_counts in state_weeklies.iterrows():
        day_lbl = week.strftime('%F')
        state_locs = [str(x) for x in
                      week_counts.index.get_level_values('State')]

        fig = px.choropleth(locations=state_locs, locationmode="USA-states",
                            title=day_lbl, scope='usa',
                            color=week_counts.values, range_color=[0, 10],
                            color_continuous_scale=['white', 'black'])

        # save the map to file and keep track of the file name
        plt_file = Path("map-plots", f"counts_{day_lbl}.png")
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
            ('seasonal_features', SeasonalTransformer(seasonal_period=52)),
            ])),
        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression())
        ])

    tscv = TimeSeriesSplit(n_splits=4)
    cali_weeklies = state_weeklies.CA
    cali_dates = cali_weeklies.index.values.reshape(-1, 1)
    cali_values = cali_weeklies.values

    real_values = list()
    pred_values = list()

    for train_index, test_index in tscv.split(cali_weeklies):
        pipeline.fit(cali_dates[train_index], cali_values[train_index])
        preds = pipeline.predict(cali_dates[test_index], to_scale=True)

        real_values += cali_values[test_index].flatten().tolist()
        pred_values += preds.flatten().tolist()

        plt.plot(cali_dates[test_index], cali_values[test_index],
                 color='black')
        plt.plot(cali_dates[test_index], preds, color='red')

    rmse_val = ((np.array(real_values)
                 - np.array(pred_values)) ** 2).mean() ** 0.5
    print(f"RMSE: {format(rmse_val, '.3f')}")


if __name__ == '__main__':
    main()
