"""Tools for downloading UFO sightings records from public datasets."""

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
                     country: str = 'usa', verbose: int = 0) -> pd.DataFrame:
    """Reading in raw data from UFO sightings website.

    Arguments
    ---------
    first_year, last_year:  the range of years (inclusive) whose sightings will
                            be considered
    country:    which nation's sightings to use
                only 'usa' and 'canada' are currently supported
    verbose:    show messages about the sightings found?

    """

    # create a table for the sightings data and only consider unique sightings
    sights_df = pd.DataFrame(sightings).drop_duplicates()

    # get valid sightings for given country
    if country == 'usa':
        sights_df = sights_df.loc[(sights_df.Country == 'USA')
                                  & sights_df.State.isin(VALID_STATES), :]

    elif country == 'canada':
        sights_df = sights_df.loc[sights_df.Country == 'Canada', :]

    else:
        raise ValueError(f"Unrecognized country for sightings: `{country}`!")

    # parse the date information into more useful format
    sights_df['Date'] = pd.to_datetime(
        [dt.split()[0] for dt in sights_df['Date']], format='%m/%d/%y')

    if verbose > 1:
        print(f"Found {sights_df.shape[0]} unique sightings!")

    return sights_df
