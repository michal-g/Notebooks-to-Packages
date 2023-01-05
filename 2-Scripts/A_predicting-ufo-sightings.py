
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
from sklearn.ensemble import (RandomForestClassifier,
                              GradientBoostingClassifier,
                              RandomForestRegressor)

from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from skits.preprocessing import ReversibleImputer
from skits.pipeline import ForecasterPipeline
from skits.feature_extraction import (AutoregressiveTransformer,
                                      SeasonalTransformer)

import imageio
from IPython.display import Image

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
    # initialize assets for scraping the reports portal
    base_url = 'https://nuforc.org/webreports'
    grab = requests.get('/'.join([base_url, 'ndxevent.html']))

    # initialize data structures for storing parsed data
    sightings = []
    col_labels = ['Date', 'City', 'State', 'Country', 'Shape', 'Duration',
                  'Summary', 'Posted', 'Images']

    # for each link to a month's data, create assets for scraping that table
    for month_link in BeautifulSoup(grab.text, 'html.parser')(
            'a', string=re.compile("[0-9]{2}\/2000")):
        month_grab = requests.get('/'.join([base_url, month_link.get('href')]))

        for row in BeautifulSoup(month_grab.text, 'html.parser')('tr'):
            cols = row.find_all('td')

            if cols:
                cur_sighting = None

                for lbl, col in zip(itertools.cycle(col_labels), cols):
                    if lbl == 'Date':
                        if cur_sighting is not None:
                            sightings.append(cur_sighting)

                        cur_sighting = {'Date': col.string}

                    else:
                        cur_sighting[lbl] = col.string

                if cur_sighting is not None:
                    sightings.append(cur_sighting)

    # create a table for the sightings data and only consider valid sightings
    sights_df = pd.DataFrame(sightings)
    sights_df = sights_df.loc[(sights_df.Country == 'USA')
                              & sights_df.State.isin(VALID_STATES), :]

    # parse the date information into more useful format, filter out duplicates
    sights_df['Date'] = pd.to_datetime(
        [dt.split()[0] for dt in sights_df['Date']],
        format='%m/%d/%y')
    sights_df = sights_df.drop_duplicates(ignore_index=True)

    state_totals = sights_df.groupby('State').size()

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])
    fig.show()

    state_dailies = sights_df.groupby(['Date', 'State']).size()
    plt_files = list()
    os.makedirs("map-plots", exist_ok=True)

    for day, day_counts in state_dailies.groupby('Date'):
        day_lbl = day.strftime('%F')
        state_locs = [str(x) for x in
                      day_counts.index.get_level_values('State')]

        fig = px.choropleth(locations=state_locs, locationmode="USA-states",
                            title=day_lbl, scope='usa',
                            color=day_counts.values, range_color=[0, 100],
                            color_continuous_scale=['white', 'black'])

        plt_file = Path("map-plots", f"counts_{day_lbl}.png")
        fig.write_image(plt_file, format='png')
        plt_files += [imageio.v2.imread(plt_file)]

    imageio.mimsave(Path("map-plots", "counts.gif"), plt_files, duration=0.03)

    pipeline = ForecasterPipeline([
        ('pre_scaler', StandardScaler()),
        ('features', FeatureUnion([
            ('ar_features', AutoregressiveTransformer(num_lags=1)),
            ('seasonal_features', SeasonalTransformer(seasonal_period=1)),
            ])),
        ('post_feature_imputer', ReversibleImputer()),
        ('post_feature_scaler', StandardScaler()),
        ('regressor', LinearRegression(fit_intercept=True))
        ])

    tscv = TimeSeriesSplit(n_splits=4)
    cali_dailies = state_dailies.loc[(slice(None), 'CA')]
    cali_dates = cali_dailies.index.get_level_values(
        'Date').values.reshape(-1, 1)
    cali_values = cali_dailies.values

    real_values = list()
    pred_values = list()

    for train_index, test_index in tscv.split(cali_dailies):
        pipeline.fit(cali_dates[train_index], cali_values[train_index])
        preds = pipeline.predict(cali_dates[test_index], to_scale=True)

        real_values += cali_values[test_index].flatten().tolist()
        pred_values += preds.flatten().tolist()

        plt.plot(cali_dates[test_index], cali_values[test_index],
                 color='black')
        plt.plot(cali_dates[test_index], preds, color='red')

    rmse_val = ((np.array(real_values) - np.array(
        pred_values)) ** 2).mean() ** 0.5
    print(f"RMSE: {format(rmse_val, '.3f')}")


if __name__ == '__main__':
    main()
