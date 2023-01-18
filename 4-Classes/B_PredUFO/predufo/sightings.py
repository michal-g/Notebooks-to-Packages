
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

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

    def __init__(self,
                 freq_sums: pd.DataFrame,
                 region: Optional[str] = None) -> None:
        self.freq_sums = freq_sums
        self.region = region

        # assets and formatted objects used by the prediction pipeline
        # scikit-learn wants Xs to be 2-dimensional and ys to be 1-dimensional
        self.tscv = TimeSeriesSplit(n_splits=4)
        self.pred_dates = self.freq_sums.index.values.reshape(-1, 1)
        self.pred_values = self.freq_sums.values

    def __str__(self):
        print(f"There are {self.freq_sums.sum()} total sightings, of which "
              f"the maximum ({self.freq_sums.max()}) took place on "
              f"{self.freq_sums.idxmax().strftime('%F')}!")

    def predict(self,
                num_lags: int, seasonal_period: int,
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

        date_values = list()
        real_values = list()
        regr_values = list()

        # for each cross-validation fold, use the training samples in the fold
        # to train the pipeline and the remaining samples to test it
        for train_index, test_index in self.tscv.split(self.freq_sums):
            pipeline.fit(self.pred_dates[train_index],
                         self.pred_values[train_index])

            preds = pipeline.predict(self.pred_dates[test_index],
                                     to_scale=True)

            # we'll keep track of the actual sightings and the predicted
            # sightings from each c-v fold for future reference
            date_values += [self.pred_dates[test_index]]
            real_values += [self.pred_values[test_index].flatten().tolist()]
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


class AmericanSightings(Sightings):

    def __init__(self, freq_sums) -> None:
        super().__init__(freq_sums.sum(axis=1))

    def __str__(self):
        print(f"There are {self.freq_sums.sum()} total sightings, of which "
              f"the maximum ({self.freq_sums.max()}) took place on "
              f"{self.freq_sums.idxmax().strftime('%F')}!")
