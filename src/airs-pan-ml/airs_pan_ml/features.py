"""Helper functions & classes to select features for model input
"""
from abc import ABC
import pandas as pd

from typing import Sequence, Union

class FeatureSet(ABC):
    """Base class for various methods of selecting features from
    a dataframe.
    """
    def subset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Given a dataframe containing input features as columns,
        return a dataframe containing only the columns that will be
        input to the model.
        """
        pass
    
    
class ColnameFeatureSet(FeatureSet):
    """Feature selector that chooses by columns with specific names.

    Parameters
    ----------
    colnames
        The columns of the dataframe to retain as model inputs.
    """
    def __init__(self, colnames: Sequence[str]):
        self.colnames = colnames
        
    def subset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.loc[:, self.colnames]
    

class PrefixSuffixFeatureSet(FeatureSet):
    """Feature selector that chooses columns with names that start
    or end with the given substrings.

    Parameters
    ----------
    prefixes
        Substrings that columns of the dataframe are allowed to start
        with. A column will be retained if it starts with any of these.
        If only one substring is desired, this may be passed as a string.

    suffixes
        Like prefixes, but for substrings at the end of the column names.
    """
    def __init__(
        self,
        prefixes: Union[str, Sequence[str]] = tuple(),
        suffixes: Union[str, Sequence[str]] = tuple()
    ):
        if len(prefixes) == 0 and len(suffixes) == 0:
            raise ValueError('Give at least one prefix or suffix')
        self.prefixes = (prefixes,) if isinstance(prefixes, str) else tuple(prefixes)
        self.suffixes = (suffixes,) if isinstance(suffixes, str) else tuple(suffixes)
        
    def subset_df(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in df.columns if c.startswith(self.prefixes) or c.endswith(self.suffixes)]
        return df.loc[:, cols]
