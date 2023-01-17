
import re
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import itertools
import pandas as pd
import plotly.express as px
from .sightings import Sightings, AmericanSightings
from abc import ABC, abstractmethod
import imageio
from typing import Optional


class _BaseSightingsDataset(ABC):

    @staticmethod
    def _load_sightings(first_year: int, last_year: int) -> pd.DataFrame:

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
        return pd.DataFrame(sightings).drop_duplicates()

    @abstractmethod
    def _filter_sightings(self, sightings: pd.DataFrame) -> pd.DataFrame:
        pass

    def _fill_daily_index(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.reindex(index=pd.date_range(f"01-01-{self.first_year}",
                                              f"12-31-{self.last_year}"),
                          fill_value=0).sort_index()

    def _create_daily_table(self) -> pd.DataFrame:
        return self._fill_daily_index(self.sightings.groupby('Date').size())

    def __init__(self,
                 first_year: int, last_year: int, verbosity: int = 0) -> None:
        self.first_year = first_year
        self.last_year = last_year
        self.verbosity = verbosity

        self.sightings = self._filter_sightings(self._load_sightings(
            first_year, last_year))

        # parse the date information into more useful format
        self.sightings['Date'] = pd.to_datetime(
            [dt.split()[0] for dt in self.sightings['Date']],
            format='%m/%d/%y'
            )

        if self.verbosity:
            print(f"Found {self.sightings.shape[0]} unique sightings!")

        self.daily_table = self._create_daily_table()

    def get_sightings(self, freq: str) -> Sightings:
        return Sightings(self.daily_table.groupby(
            pd.Grouper(freq=freq)).sum().astype(int))


class AmericanSightingsDataset(_BaseSightingsDataset):

    # we will be very careful to filter sightings that can be mapped to states
    VALID_STATES = {
        'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
        'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
        'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
        'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
        'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
        }

    def _filter_sightings(self, sightings: pd.DataFrame) -> pd.DataFrame:
        valid_sights = sightings.State.isin(self.VALID_STATES)
        usa_sights = sightings.loc[(sightings.Country == 'USA')
                                   & valid_sights, :]

        return usa_sights

    def _create_daily_state_table(self) -> pd.DataFrame:
        return self._fill_daily_index(self.sightings.groupby(
            ['Date', 'State']).size().unstack().fillna(0))

    def __init__(self,
                 first_year: int, last_year: int, verbosity: int = 0) -> None:
        super().__init__(first_year, last_year, verbosity)
        self.daily_state_table = self._create_daily_state_table()

    def get_state_sightings(
            self,
            freq: str, states: Optional[list[str]] = None
            ) -> AmericanSightings:
        if states:
            daily_table = self.daily_state_table.loc[:, list(states)]
        else:
            daily_table = self.daily_state_table.copy()

        return AmericanSightings(daily_table.groupby(
            pd.Grouper(axis=0, freq=freq)).sum().astype(int))

    def plot_totals_map(self) -> None:
        """Mapping state totals across entire time period."""
        state_totals = self.sightings.groupby('State').size()

        fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                            scope="usa", locationmode="USA-states",
                            color=state_totals.values,
                            range_color=[0, state_totals.max()],
                            color_continuous_scale=['white', 'red'])

    def animate_totals_map(self) -> None:
        """Animating periodical state totals."""
        plt_files = list()

        # create a map of sightings by state for each time period
        for dt, freq_counts in self.daily_state_table.iterrows():
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


class CanadianSightingsDataset(_BaseSightingsDataset):

    def _filter_sightings(self, sightings: pd.DataFrame) -> pd.DataFrame:
        return sightings.loc[sightings.Country == 'Canada', :]
