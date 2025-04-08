from abc import ABC, abstractmethod
from enum import Enum
import netCDF4 as ncdf
from string import ascii_lowercase

from typing import Any, Sequence, Union
    

class DeprecatedError(Exception):
    pass


class AbstractLabeler(ABC):
    @abstractmethod
    def label_axes(self, ax, index):
        """Add the label corresponding to zero-based index to ``ax``"""

    @abstractmethod
    def get_label(self, index):
        """Get the label corresponding to zero-based ``index``"""


class LetterLabeler:
    """Label an axes with a letter based on the plot index

    Parameters
    ----------
    xpos
        x-position of the label in axes-relative coordinates (i.e. 0 is left edge, 1 is right edge)

    ypos
        y-position of the label in axes-relative coordinates (i.e. 0 is bottom edge, 1 is top edge)

    repeat_method
        How to label indices > 25. Options are:

        * "cyc" (default) gives 25 = "z", 26 = "aa", 27 = "ab", 51 = "az", 52 = "ba", etc.
        * "dup" gives 25 = "z", 26 = "aa", 27 = "bb", 51 = "zz", 52 = "aaa", etc.

        Note that this argument can be passed a string or a variant of :class:`LetterLabeler.RepMethod`.
    """
    class RepMethod(Enum):
        DUPLICATE = 'dup'
        CYCLE = 'cyc'

    def __init__(self, xpos: float = -0.1, ypos: float = 1.05, repeat_method='cyc') -> None:
        self._xpos = xpos
        self._ypos = ypos
        self._rep_method = self.RepMethod(repeat_method)
        
    def label_axes(self, ax, index: int) -> None:
        """Label ``ax`` with the letter corresponding to zero-based ``index``.
        
        If ``index >= 26``, the letter is doubled, tripled, etc. as necessary.
        """
        c = self.get_label(index)
        ax.text(self._xpos, self._ypos, f'({c})', transform=ax.transAxes)
        
    def get_label(self, index: int) -> str:
        """Get the letter corresponding to zero-based ``index``.
        
        If ``index >= 26``, the letter is doubled, tripled, etc. as necessary.
        """
        if self._rep_method == self.RepMethod.DUPLICATE:
            n = index // 26 + 1
            i = index % 26
            return ascii_lowercase[i]*n
        elif self._rep_method == self.RepMethod.CYCLE:
            letter_inds = []
            n = index
            while True:
                letter_inds.append(n % 26)
                if n < 26:
                    break
                n = (n // 26) - 1
            return ''.join(ascii_lowercase[i] for i in reversed(letter_inds))



def find_attribute(nc_object: Union[ncdf.Dataset, ncdf.Group, ncdf.Variable], possible_names: Sequence[str], return_value: bool = False) -> Union[str, Any]:
    """Find an attribute on a netCDF object that may have one of several names

    Parameters
    ----------
    nc_object
        Handle to the netCDF dataset, group, or variable to search for the attribute on.

    possible_names
        An list or similar of names that the attribute might have. Note that case is ignored
        when searching for the attribute.

    return_value
        If ``False``, then the name of the attribute is returned. If ``True``, the value of
        the attribute is.

    Returns
    -------
    attr
        Either the name or value of the attribute, depending on the value of ``return_value``.

    Raises
    ------
    AttributeError
        If no attribute is found on ``nc_object`` matching any of the ``possible_names`` (ignoring case).
    """
    lower_case_names = set(n.lower() for n in possible_names)
    for attr in nc_object.ncattrs():
        if attr.lower() in lower_case_names:
            if return_value:
                return getattr(nc_object, attr)
            else:
                return attr

    raise AttributeError(f'Could not find units attribute on {nc_object.name}')