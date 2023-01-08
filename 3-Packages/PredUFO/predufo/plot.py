
import numpy as np
import pandas as pd

from pathlib import Path
import imageio
import plotly.express as px
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (14, 9)


def plot_totals_map(sights_df: pd.DataFrame) -> None:
    """Mapping state totals across entire time period."""
    state_totals = sights_df.groupby('State').size()

    fig = px.choropleth(locations=[str(x) for x in state_totals.index],
                        scope="usa", locationmode="USA-states",
                        color=state_totals.values,
                        range_color=[0, state_totals.max()],
                        color_continuous_scale=['white', 'red'])

    fig.write_image(Path("map-plots", "state-totals.png"), format='png')


def animate_totals_map(weeklies: pd.DataFrame) -> None:
    """Animating weekly state totals."""
    plt_files = list()

    # create a map of sightings by state for each week
    for week, week_counts in weeklies.iterrows():
        day_lbl = week.strftime('%F')
        state_locs = [str(x) for x in
                      week_counts.index.get_level_values('State')]

        fig = px.choropleth(locations=state_locs,
                            locationmode="USA-states",
                            title=day_lbl, scope='usa',
                            color=week_counts.values, range_color=[0, 10],
                            color_continuous_scale=['white', 'black'])

        plt_file = Path("map-plots", "gif-comps", f"counts_{day_lbl}.png")
        fig.write_image(plt_file, format='png')
        plt_files += [imageio.v2.imread(plt_file)]

    # create an animation using the individual weekly maps
    imageio.mimsave(Path("map-plots", "counts.gif"), plt_files,
                    duration=0.03)


def plot_predictions(date_values: list[np.array], real_values: list[np.array],
                     regr_values: list[np.array]) -> None:
    """Plotting predicted sighting counts versus historical counts. """
    fig, ax = plt.subplots(figsize=(10, 6))

    for dates, reals, regrs in zip(date_values, real_values, regr_values):
        ax.plot(dates, reals, color='black')
        ax.plot(dates, regrs, color='red')

    fig.savefig(Path("map-plots", "predictions.png"),
                bbox_inches='tight', format='png')
