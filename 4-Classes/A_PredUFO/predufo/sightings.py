
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import itertools
from typing import Optional, Iterable
from .utils import get_states_lbl

import numpy as np
import pandas as pd

import imageio
import plotly.express as px
import matplotlib.pyplot as plt

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


class Sightings:
    """Total UFO sightings across a range of time windows."""

    def __init__(self,
                 freq_table: pd.DataFrame, freq: str, country: str) -> None:
        """
        Arguments
        ---------
        freq_table: a day x region table of sightings
        freq:       a time frequency to use when calculating the sighting sums
                    any offset alias supported by pandas is supported here
        country:    which country the sightings occurred in

        """
        if len(freq_table.shape) == 2:
            freq_sums = freq_table.groupby(
                pd.Grouper(axis=0, freq=freq, sort=True)).sum().astype(int)

        elif len(freq_table.shape) == 1:
            freq_sums = freq_table.groupby(
                pd.Grouper(freq=freq)).sum().astype(int)

        else:
            raise ValueError("Unrecognized shape for sightings table!")

        self.freq_sums = freq_sums
        self.country = country

    def animate_totals_map(self) -> None:
        """Animating periodical state totals."""
        if self.country == 'usa':
            print("warning: state total animations only available "
                  "for US sightings!")
            return None

        plt_files = list()

        # create a map of sightings by state for each time period
        for dt, freq_counts in self.freq_sums.iterrows():
            day_lbl = dt.strftime('%F')
            state_locs = [str(x) for x in
                          freq_counts.index.get_level_values('State')]

            fig = px.choropleth(locations=state_locs,
                                locationmode="USA-states",
                                title=day_lbl, scope='usa',
                                color=freq_counts.values, range_color=[0, 10],
                                color_continuous_scale=['white', 'black'])

            plt_file = Path("map-plots", "gif-comps", f"counts_{day_lbl}.png")
            fig.write_image(plt_file, format='png')
            plt_files += [imageio.v2.imread(plt_file)]

        # create an animation using the individual weekly maps
        imageio.mimsave(Path("map-plots", "counts.gif"), plt_files,
                        duration=0.03)

    def predict(self,
                num_lags: int, seasonal_period: int,
                states: Optional[Iterable[str]] = None,
                create_plots: bool = False,
                verbose: int = 0) -> tuple[list[float], float]:
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

        if states:
            pred_byfreq = self.freq_sums.loc[:, list(states)].sum(axis=1)
        else:
            pred_byfreq = self.freq_sums.copy()

        # assets and formatted objects used by the prediction pipeline
        # scikit-learn wants Xs to be 2-dimensional and ys to be 1-dimensional
        tscv = TimeSeriesSplit(n_splits=4)
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

        # for each cross-validation fold, use the training samples in the fold
        # to train the pipeline and the remaining samples to test it
        for train_index, test_index in tscv.split(pred_byfreq):
            pipeline.fit(pred_dates[train_index], pred_values[train_index])
            preds = pipeline.predict(pred_dates[test_index], to_scale=True)

            # we'll keep track of the actual sightings and the predicted
            # sightings from each c-v fold for future reference
            date_values += [pred_dates[test_index]]
            real_values += [pred_values[test_index].flatten().tolist()]
            regr_values += [preds.flatten().tolist()]

        # plot predicted sighting counts versus historical counts
        if create_plots:
            fig, ax = plt.subplots(figsize=(10, 6))

            for dates, reals, regrs in zip(date_values, real_values,
                                           regr_values):
                ax.plot(dates, reals, color='black')
                ax.plot(dates, regrs, color='red')

            fig.savefig(Path("map-plots", "predictions.png"),
                        bbox_inches='tight', format='png')

        # measure the quality of the predictions using root-mean-squared-error
        rmse_val = ((np.array(real_values)
                     - np.array(regr_values)) ** 2).mean() ** 0.5

        return regr_values, rmse_val


class SightingsDataset:
    """A repository of downloaded UFO sightings records.

    Attributes
    ---------
    first_year, last_year (int):  the range of years (inclusive) whose
                                  sightings will be considered
    country (str):  which nation's sightings to use
                    only 'usa' and 'canada' are currently supported
    verbosity (int):    show messages about the sightings found?
    self.sights_data (pd.DataFrame): a table of downloaded sightings, with one
                                     row per sighting
    """

    # we will be very careful to filter sightings that can be mapped to states
    VALID_STATES = {
        'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
        'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
        'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
        'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
        }

    def __init__(self,
                 first_year: int, last_year: int, country: str = 'usa',
                 verbosity: int = 0) -> None:

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

        # for each link to a month's table, create assets for scraping it
        for month_link in BeautifulSoup(grab.text, 'html.parser')(
                'a', string=re.compile(f"[0-9]{{2}}\/{year_regex}")):
            month_grab = requests.get(
                '/'.join([base_url, month_link.get('href')]))

            # the HTML formatting is kind of weird; we first grab the outermost
            # of a recursively defined set of table elements
            table_data = BeautifulSoup(
                month_grab.text, 'html.parser')('tr')[1]('td')
            cur_sighting = None

            # then we loop over a set of table entries that are defined as one
            # big row??? maybe there's an easier way to do this but that's ok
            for lbl, col in zip(itertools.cycle(col_labels), table_data):
                if lbl == 'Date':
                    if cur_sighting is not None:
                        sightings.append(cur_sighting)

                    cur_sighting = {'Date': col.string}

                # start a new sighting record, after adding the last record to
                # the list of sightings if this is not the first row
                else:
                    cur_sighting[lbl] = col.string

            # accounting for the last row
            if cur_sighting is not None:
                sightings.append(cur_sighting)

        # create a table for the sightings and only consider unique sightings
        sights_df = pd.DataFrame(sightings).drop_duplicates()

        # get valid sightings for given country
        if country == 'usa':
            valid_sights = sights_df.State.isin(self.VALID_STATES)
            sights_df = sights_df.loc[(sights_df.Country == 'USA')
                                      & valid_sights, :]

        elif country == 'canada':
            sights_df = sights_df.loc[sights_df.Country == 'Canada', :]

        else:
            raise ValueError(
                f"Unrecognized country for sightings: `{country}`!")

        # parse the date information into more useful format
        sights_df['Date'] = pd.to_datetime(
            [dt.split()[0] for dt in sights_df['Date']], format='%m/%d/%y')

        if verbosity:
            print(f"Found {sights_df.shape[0]} unique sightings!")

        self.first_year = first_year
        self.last_year = last_year
        self.country = country
        self.verbosity = verbosity
        self.sights_data = sights_df

    def get_sightings(self, freq: str) -> Sightings:
        if self.country == 'usa':
            freq_table = self.sights_data.groupby(
                ['Date', 'State']).size().unstack().fillna(0)

            freq_table = freq_table.reindex(
                index=pd.date_range(f"01-01-{self.first_year}",
                                    f"12-31-{self.last_year}"),
                fill_value=0
                ).sort_index()

        elif self.country == 'canada':
            freq_table = self.sights_data.groupby('Date').size().reindex(
                index=pd.date_range(f"01-01-{self.first_year}",
                                    f"12-31-{self.last_year}"),
                fill_value=0
                ).sort_index()

        return Sightings(freq_table, freq, self.country)

    def plot_totals_map(self) -> None:
        """Mapping state totals across entire time period."""
        state_totals = self.sights_data.groupby('State').size()

        fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                            scope="usa", locationmode="USA-states",
                            color=state_totals.values,
                            range_color=[0, state_totals.max()],
                            color_continuous_scale=['white', 'red'])
