import pandas as pd
from scipy import stats
from scipy.stats._result_classes import PearsonRResult


def pearsonr(sample1: pd.Series, sample2: pd.Series) -> PearsonRResult:
    """Just a thin wrapper around pearsonr that can handle nans.
    Just like the nan_policy='omit' option in scipy.stats.spearmanr
    """
    if sample1.isna().any() or sample2.isna().any():
        print(
            "Pearson's r is not defined for unequal sample sizes. Dropping observations with missing samples."
        )
        df = pd.concat([sample1, sample2], axis=1).dropna()
        sample1 = df.iloc[:, 0]
        sample2 = df.iloc[:, 1]
    return stats.pearsonr(sample1, sample2)


def cohens_d(sample1: pd.Series, sample2: pd.Series) -> float:
    if sample1.isna().any() or sample2.isna().any():
        print(
            "Cohen's D is not defined for unequal sample sizes. Dropping observations with missing samples."
        )
        df = pd.concat([sample1, sample2], axis=1).dropna()
        sample1 = df.iloc[:, 0]
        sample2 = df.iloc[:, 1]
    return abs(sample1.mean() - sample2.mean()) / (sample1 - sample2).std()


def interpret_cohens_d(cohens_d):
    """
    Determines text interpretation of effect size given Cohen's d value

    :param cohens_d: float of Cohen's d value
    :returns: effect_size_interpretation: adjective to describe magnitude of effect size
    """
    # https://dfrieds.com/math/effect-size.html
    if 0 <= cohens_d < 0.1:
        effect_size_interpretation = "Very Small"
    elif 0.1 <= cohens_d < 0.35:
        effect_size_interpretation = "Small"
    elif 0.35 <= cohens_d < 0.65:
        effect_size_interpretation = "Medium"
    elif 0.65 <= cohens_d < 0.9:
        effect_size_interpretation = "Large"
    elif cohens_d >= 0.9:
        effect_size_interpretation = "Very Large"
    return effect_size_interpretation
