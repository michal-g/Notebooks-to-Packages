
import numpy as np
import pandas as pd
from typing import Optional, Iterable

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

from .plot import plot_predictions
from .utils import get_states_lbl


def predict_sightings(
        sightings: pd.DataFrame, num_lags: int, seasonal_period: int,
        states: Optional[Iterable[str]] = None,
        create_plots: bool = False, verbose: int = 0
        ) -> tuple[list[float], float]:
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
        pred_byfreq = sightings.loc[:, list(states)].sum(axis=1)
    else:
        pred_byfreq = sightings.copy()

    # assets and specially formatted objects used by the prediction pipeline
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

    # for each cross-validation fold, use the training samples in the fold to
    # train the pipeline and the remaining samples to test it
    for train_index, test_index in tscv.split(pred_byfreq):
        pipeline.fit(pred_dates[train_index], pred_values[train_index])
        preds = pipeline.predict(pred_dates[test_index], to_scale=True)

        # we'll keep track of the actual sightings and the predicted sightings
        # from each c-v fold for future reference
        date_values += [pred_dates[test_index]]
        real_values += [pred_values[test_index].flatten().tolist()]
        regr_values += [preds.flatten().tolist()]

    if create_plots:
        plot_predictions(date_values, real_values, regr_values)

    # measure the quality of the predictions using root-mean-squared-error
    rmse_val = ((np.array(real_values)
                 - np.array(regr_values)) ** 2).mean() ** 0.5

    return regr_values, rmse_val
