"""Tools for downloading UFO sightings records from public datasets."""

import os
import pandas as pd


# we will be very careful to filter sightings that can be mapped to states
VALID_STATES = {
    'AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
    'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
    'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
    'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX',
    'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'
    }

VALID_PROVINCES = {
    'ON', 'MB', 'BC', 'AB', 'PQ', 'SK', 'NB', 'NS', 'NF', 'YT', 'NT', 'PE'
    }


def parse_sightings(first_year: int, last_year: int,
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

    sights_df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "nuforc_events_complete.csv"),
        usecols=['event_time', 'city', 'state', 'shape', 'duration', 'summary']
        )

    # get valid sightings for given country
    if country == 'usa':
        sights_df = sights_df.loc[sights_df.state.isin(VALID_STATES), :]

    elif country == 'canada':
        sights_df = sights_df.loc[sights_df.state.isin(VALID_PROVINCES), :]

    else:
        raise ValueError(f"Unrecognized country for sightings: `{country}`!")

    # parse the date information into more useful format
    sights_df['event_time'] = pd.to_datetime(
        sights_df.event_time, format="%Y-%m-%dT%H:%M:%SZ", errors='coerce')
    sights_df = sights_df.loc[~sights_df['event_time'].isna(), :]

    yearly_sightings = sights_df.groupby(
        [sights_df.event_time.dt.year, 'state']).size().unstack().fillna(0)
    yearly_sightings.reindex(range(first_year, last_year + 1), fill_value=0.)

    if verbose > 1:
        print(f"Found {yearly_sightings.values.sum()} unique sightings!")

    return yearly_sightings
