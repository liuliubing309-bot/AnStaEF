from typing import List
import numpy as np
import pandas as pd
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        pass

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.isocalendar().week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    features_by_offsets = {
        offsets.YearEnd: [],
        offsets.MonthEnd: [MonthOfYear],
        offsets.Week: [MonthOfYear, WeekOfYear],
        offsets.Day: [MonthOfYear, DayOfMonth, DayOfWeek],
        offsets.BusinessDay: [MonthOfYear, DayOfMonth, DayOfWeek],
        offsets.Hour: [MonthOfYear, DayOfMonth, DayOfWeek, HourOfDay],
        offsets.Minute: [MonthOfYear, DayOfMonth, DayOfWeek, HourOfDay, MinuteOfHour],
        offsets.Second: [
            MonthOfYear,
            DayOfMonth,
            DayOfWeek,
            HourOfDay,
            MinuteOfHour,
            SecondOfMinute,
        ],
    }

    offset = to_offset(freq_str)

    for offset_type, feature_classes in features_by_offsets.items():
        if isinstance(offset, offset_type):
            return [cls() for cls in feature_classes]

    supported_freq_msg = f"""
    Unsupported frequency: {freq_str}
    The following frequencies are supported:
        Y   - yearly
        M   - monthly
        W   - weekly
        D   - daily
        B   - business days
        H   - hourly
        T   - minutely
        S   - secondly
    """
    raise RuntimeError(supported_freq_msg)


def time_features(dates, timeenc=1, freq='h'):
    if timeenc == 0:
        return None

    feature_list = time_features_from_frequency_str(freq)

    features = [ft(dates) for ft in feature_list]

    return np.vstack(features).transpose()
