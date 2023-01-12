
import re
import requests
from bs4 import BeautifulSoup
import pandas as pd
import itertools


# we will be very careful to filter sightings that can be mapped to states
VALID_STATES = {
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
    'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
    }


def scrape_sightings(first_year: int, last_year: int,
                     verbose: int) -> pd.DataFrame:
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
