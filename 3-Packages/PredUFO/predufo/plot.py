"""Plots describing sightings data and evaluating prediction performance."""

import numpy as np
import pandas as pd

from pathlib import Path
import imageio
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14, 9)


def plot_totals_map(sights_df: pd.DataFrame) -> None:
    """Mapping state totals across entire time period.

    Arguments
    ---------
    sights_df:  table of individual sightings, one per row

    """
    state_totals = sights_df.groupby('State').size()

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])

    fig.write_image(Path("map-plots", "state-totals.png"), format='png')


def animate_totals_map(sightings: pd.DataFrame) -> None:
    """Plotting a gif of periodical state totals.

    Arguments
    ---------
    sightings:  period x state table of sighting counts across time windows

    """
    plt_files = list()

    # create a map of sightings by state for each time period
    for dt, freq_counts in sightings.iterrows():
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


def plot_predictions(date_values: list[np.array], real_values: list[np.array],
                     regr_values: list[np.array]) -> None:
    """Plotting predicted sighting counts versus historical counts.

    The arguments to this function are lists of indices, Xs, and ys generated
    during learning cross-validation: thus the ith element of each list
    corresponds to the testing values returned from the ith c-v fold.

    Arguments
    ---------
    date_values:    the date indices used for these sightings
    real_values:    the historical sighting counts for these dates
    regr_values:    the sighting counts predicted by our regressor

    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for dates, reals, regrs in zip(date_values, real_values, regr_values):
        ax.plot(dates, reals, color='black')
        ax.plot(dates, regrs, color='red')

    fig.savefig(Path("map-plots", "predictions.png"),
                bbox_inches='tight', format='png')
