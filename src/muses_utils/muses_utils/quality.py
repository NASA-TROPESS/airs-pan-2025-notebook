"""Tools to filter MUSES data on quality metrics and apply bias corrections.
"""

from abc import ABC, abstractmethod
from enum import Enum
import cartopy.crs as ccrs
from functools import cache, partial
import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as ncdf
import pandas as pd
from pathlib import Path
import pickle
from scipy.optimize import least_squares
from warnings import warn
import xarray as xr

from jllutils import plots as jplt
from jllutils import stats as jstats
from jllutils import miscutils
from muses_utils import readers, conversions

from typing import Callable, Optional, Sequence, Union


class RecalcMode(Enum):
    NEVER = 'never'
    IF_MISSING = 'if_missing'
    IF_NEEDED = 'if_needed'
    ALWAYS = 'always'


class AbstractQualityFilterer(ABC):
    """Base class for filtering MUSES data on quality

    Child classes will need to implement:

    * ``quality_flags`` - a method that returns a boolean numpy array (1 = good, 0 = bad) for filtering soundings
    * ``sigma_out_of_bounds`` - a method that returns a metric for out out-of-bounds a value was
    * ``reason`` - a method that returns a vector of reasons why each sounding was flagged
    * ``get_plotting_df`` - a method that returns a dataframe with the quantities used to filter on
    * ``plot`` - a default plot of the quantities filtered on

    All methods take a path to a standard MUSES level 2 file. This is intended to operate on combined files, but in
    theory could work on single sounding files (though that has not been tested). Child classes that need other files
    (e.g. radiance) should derive that file name from the L2 file.
    """

    @abstractmethod
    def quality_flags(self, l2_file: str):
        """Return a numpy vector that is ``True`` for good soundings and ``False`` for bad ones.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        flags
            A numpy boolean vector; ``True`` = good quality, ``False`` = bad quality.
        """

    @abstractmethod
    def sigma_out_of_bounds(self, l2_file: str):
        """Return a numpy vector that gives a normalized amount by which a given flagged sounding is "out-of-bounds"

        Good quality soundings will have 0. Values may be negative. The intention is that, for a flag based on one variable,
        the return value will be the difference between the value and the upper or lower allowed limit (whichever is nearer),
        divided by the allowed range (i.e. upper limit minus lower limit). For more complicated filtering, other implementations
        will be needed. However, since this is used to determine which quality flag is the most severe, care should be taken to
        make sure that values are comparable across multiple filtering criteria (i.e. 2 for any criterion is more out-of-bounds
        than 1 for any other criterion).

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        sigmas
            A numpy float vector that describes how "out of bounds" a given sounding is. Good quality soundings
            will have a value of 0.
        """

    @abstractmethod
    def reason(self, l2_file: str) -> np.ndarray:
        """Return a string or vector of strings giving the reason why a sounding was flagged

        For most child classes, the returned vector will have only two unique strings: the empty string,
        and a string that represents the filtering cause, e.g. "radianceResidualRMS_out_of_bounds". The possibility
        that a filter may have multiple reasons is left open; normally this would be used only in the class
        that combines all the "for master" filters to produce the master quality flag, but users should *not* assume
        that other classes will never have multiple reasons.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        reasons
            A vector containing the reason each sounding was flagged. Soundings not flagged
            will have an empty string.
        """

    @abstractmethod
    def get_plotting_df(self, l2_file: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        criteria
            A dataframe which will have a longitude and latitude as values in its row index, and each
            variable or quantity used to filter on as a column.
        """
        pass

    def plot(self, l2_file: str, axs=None, **scatter_kws):
        """
        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        axs
            The matplotlib axes or sequence of axes to plot into. By default, this may be either an instance of 
            matplotlib axes directly, or a sequence containing said axes. In the latter case, the first element is 
            always used.

        scatter_kws
            Additional keyword arguments to pass to the :func:`~matplotlib.pyplot.scatter` plotting function.
            The following keywords are set by default (but can be overridden):

            * ``s = 6``
            * ``vmin`` and ``vmax`` will be set to the lower and upper allowed limits, respectively
            * ``cmap`` is set to a "viridis" colormap, with "blue" and "red" as the under and over colors, respectively.
              The colorbar will set the ``extend`` keyword intelligently to include the under/over arrows as needed.

        Returns
        -------
        axs
            The axes or sequence of axes plotted into.

        Notes
        -----
        The default implementation will use the :meth:`get_plotting_df` and :meth:`quality_flags` methods to get the
        data to plot and the number of flagged soundings, respectively. Child classes can customize what is plotted
        by either how those methods are implemented, or by overriding this method, loading a dataframe and flags
        differently, and passing those along to the :meth:`_plot_inner` function. That function will always use the
        first column of the dataframe, so if a different column is desired, you will need to modify the dataframe 
        passed into :meth:`_plot_inner`. If :meth:`_plot_inner` gets a dataframe with >1 column, it issues a warning.

        If a wholly custom implementation is needed, child classes implementing this function are recommended to follow 
        these conventions for consistency:

        * The name of the variable plotted is added as a label to the colorbar
        * The number of flagged soundings, total soundings, and percent of flagged soundings is included in the title
        * Axes default to the :class:`cartopy.crs.PlateCarree` projection.
        * Gridlines, x, and y labels are *not* drawn by default.

        If your instance has attributes ``_minval`` or ``_maxval``, those will be set as the default values for "vmin" 
        and "vmax" in the default implementation's scatter plot keywords.
        """
        df = self.get_plotting_df(l2_file)
        flags = self.quality_flags(l2_file)
        return self._plot_inner(df, flags, axs=axs, **scatter_kws)

    def _plot_inner(self, df: pd.DataFrame, flags: np.ndarray, axs=None, **scatter_kws):
        """Helping plotting function that does the default plots.

        Parameters
        ----------
        df
            Dataframe with lon and lat as part of a multiindex and the data to plot as the first column.

        flags
            Array of quality flags, 1 = good, 0 = bad

        axs
            The matplotlib axes or sequence of axes to plot into. This may be either an instance of matplotlib
            axes directly, or a sequence containing said axes. In the latter case, the first element is always
            used.

        scatter_kws
            Additional keyword arguments to pass to the :func:`~matplotlib.pyplot.scatter` plotting function.
            The following keywords are set by default (but can be overridden):

            * ``s = 6``
            * ``vmin`` and ``vmax`` will be set to the lower and upper allowed limits, respectively
            * ``cmap`` is set to a "viridis" colormap, with "blue" and "red" as the under and over colors, respectively.
              The colorbar will set the ``extend`` keyword intelligently to include the under/over arrows as needed.
        """
        if df.shape[1] != 1:
            warn(f'Multiple columns in dataframe, will use {df.columns[0]}')
        nflagged = np.sum(flags == 0)
        ntot = flags.size
        lon = df.index.get_level_values('lon')
        lat = df.index.get_level_values('lat')
        vals = df.iloc[:,0]

        scatter_kws.setdefault('s', 6)
        scatter_kws.setdefault('cmap', jplt.colormap_out_of_bounds(under='blue', over='red'))
        if hasattr(self, '_minval'):
            scatter_kws.setdefault('vmin', self._minval)
        if hasattr(self, '_maxval'):
            scatter_kws.setdefault('vmax', self._maxval)

        if axs is None:
            _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
        else:
            try:
                ax = axs[0]
            except TypeError:
                # Assume that this means axes were passed directly, not in a list
                ax = axs

        h = ax.scatter(lon, lat, c=vals, **scatter_kws)
        extend = jplt.cbextend(vals, vmin=scatter_kws.get('vmin'), vmax=scatter_kws.get('vmax'))
        cb = plt.colorbar(h, ax=ax, label=df.columns[0], extend=extend)
        ax.set_title(f'{nflagged}/{ntot} ({nflagged/ntot*100:.2f}%) flagged')
        return ax, cb


class AbstractFilterWithCache(AbstractQualityFilterer):
    @property
    def recalc_mode(self) -> RecalcMode:
        return self._recalc

    @abstractmethod
    def _find_cache_file(self, l2_file: str) -> Path:
        """Get the path to use for the cache file
        """
        pass

    @abstractmethod
    def _is_cache_var_missing(self, cache_ds: ncdf.Dataset) -> bool:
        """Given the cache dataset, return ``True`` if it is missing required variables
        """
        pass

    @abstractmethod
    def _are_cache_attrs_wrong(self, cache_ds: ncdf.Dataset) -> bool:
        """Given the cache dataset, return ``True`` if it is has attributes that do not match this class
        """
        pass

    @abstractmethod
    def _load_from_cache(self, l2_file: str):
        """Load the quality filter variables from the cache.

        This can return them in whatever form the class prefers, usually a numpy array or Pandas dataframe.
        """
        pass

    @abstractmethod
    def _save_to_cache(self, l2_file: str, cache_data: dict):
        """Save whatever data is needed to the cache file.

        Note that this implementation must ensure any attributes required by ``_are_cache_attrs_wrong``
        are written as well.
        """
        pass

    def _copy_id_vars(self, l2_file: str, cache_ds: ncdf.Dataset):
        with ncdf.Dataset(l2_file) as l2_ds:
            nsnd = l2_ds['Latitude'].size
            nschar = l2_ds['SoundingID'].shape[1]
            sid_dim_2 = f'string{nschar}'
            cache_ds.createDimension('Grid_Targets', nsnd)
            cache_ds.createDimension(sid_dim_2, nschar)

            for varname in ['SoundingID', 'Latitude', 'Longitude']:
                dims = ['Grid_Targets', sid_dim_2] if varname == 'SoundingID' else ['Grid_Targets']
                var = cache_ds.createVariable(varname, l2_ds[varname].dtype, dims)
                var[:] = l2_ds[varname][:]
                attrs = {a: getattr(l2_ds[varname], a) for a in l2_ds[varname].ncattrs()}
                var.setncatts(attrs)


    def is_recalc_needed(self, l2_file) -> bool:
        """Check if the criteria need recalculated, return True if so, False if they can be read from the cache file.

        Raises a RuntimeError if not allowed to recalculate, but the cache file is missing or has different settings/sounding IDs.
        """
        if self.recalc_mode == RecalcMode.ALWAYS:
            return True

        cache_file = self._find_cache_file(l2_file)
        missing = not cache_file.exists()
        if missing:
            sid_mismatch = True
            var_missing = True
            attr_mismatch = True
        else:
            with ncdf.Dataset(l2_file) as l2_ds, ncdf.Dataset(cache_file) as c_ds:
                l2_sids = l2_ds['SoundingID'][:].filled(-999)
                cache_sids = c_ds['SoundingID'][:].filled(-999)
                var_missing = self._is_cache_var_missing(c_ds)
                sid_mismatch = not np.array_equal(l2_sids, cache_sids)
                attr_mismatch = self._are_cache_attrs_wrong(c_ds)

        if self.recalc_mode == RecalcMode.NEVER:
            if missing or var_missing or sid_mismatch or attr_mismatch:
                raise RuntimeError(f'Recalculating the AIRS PAN criteria not allowed, but either the cache file is missing ({missing}), the variable is missing ({var_missing}), or the SIDs do not match the L2 file ({sid_mismatch})')
        elif self.recalc_mode == RecalcMode.IF_MISSING:
            if sid_mismatch:
                raise RuntimeError('Recalculating the AIRS PAN criteria only allowed if the cache file is missing, but the SIDs do not match the L2 file')
            elif attr_mismatch:
                raise RuntimeError(f'Mismatch between instance and cached min_freq ({self._min_freq} vs {c_ds.min_freq}) and/or near zero cutoff ({self._near_zero_cutoff} vs {c_ds.near_zero_cutoff})')
            else:
                return missing or var_missing
        elif self.recalc_mode == RecalcMode.IF_NEEDED:
            return missing or var_missing or sid_mismatch or attr_mismatch
        else:
            raise NotImplementedError(f'Recalculation mode {self.recalc_mode}')


class StdFlagQualityFilterer(AbstractQualityFilterer):
    """Quality filtering class that uses an existing MUSES quality flag variable.

    Parameters
    ----------
    flag_var
        Name of the quality flag variable in the combined L2/L2_Lite file. Only
        soundings with this value = 1 are considered good quality.

    alternate_step
        If given, then the quality flag will be read from the combined MUSES file for
        this step rather than the level 2 file directly passed in. For example, if the
        L2 file was `Lite_Products_L2-PAN-0.nc`, then passing "H2O-1" for this argument
        would read the quality flag from `Lite_Products_L2-H2O-1.nc` instead.

    alt_file_fxn
        If `alternate_step` cannot be used, then pass to this argument a function that 
        takes a string which is the path to the L2 file and returns a `Path` to the
        file to read the quality flags from.
    """
    def __init__(self, flag_var: str, alternate_step: Optional[str] = None, alt_file_fxn: Optional[Callable[[str], Path]] = None):
        self._flag_var = flag_var
        if alt_file_fxn is not None and alternate_step is not None:
            raise TypeError('Provide only one of `alternate_step` or `alt_file_fxn`')
        elif alternate_step is not None:
            self._alt_file_fxn = partial(_find_related_step_default, desired_step=alternate_step)
        elif alt_file_fxn is not None:
            self._alt_file_fxn = alt_file_fxn
        else:
            self._alt_file_fxn = None

    def quality_flags(self, l2_file: str):
        src_file = self._get_source_file(l2_file)
        with ncdf.Dataset(src_file) as ds:
            varpath = readers.find_var_at_path_ignore_case(ds, self._flag_var)
            flags = readers.get_nc_var(ds, varpath)
        return flags == 1

    def sigma_out_of_bounds(self, l2_file):
        good_soundings = self.quality_flags(l2_file)
        return 1 - good_soundings

    def reason(self, l2_file):
        good = self.quality_flags(l2_file)
        reasons = np.full(good.size, '', dtype=object)
        reasons[good == 0] = f'Quality flag {self._flag_var} is 0'
        return reasons

    def get_plotting_df(self, l2_file):
        src_file = self._get_source_file(l2_file)
        with ncdf.Dataset(src_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array'),
                self._flag_var: readers.find_nc_var(ds, self._flag_var, return_type='array')
            }

        return pd.DataFrame(data).set_index(['lon', 'lat'])

    def _get_source_file(self, l2_file: str) -> Path:
        if self._alt_file_fxn is None:
            return Path(l2_file)
        else:
            return self._alt_file_fxn(l2_file)


class VariableQualityFilterer(AbstractQualityFilterer):
    """Quality filtering class that filters on a single variable

    Parameters
    ----------
    varname
        Name of the variable in the L2 combined file to filter on. It will be searched for through all
        groups in the file and ignoring case, so you only need to specify the variable name.

    minval
        Minimum allowed value for this metric.

    maxval
        Maximum allowed value for this metric.
    """
    def __init__(self, varname: str, minval: float, maxval: float):
        self._varname = varname
        self._minval = minval
        self._maxval = maxval

    def __repr__(self):
        return f'<VariableQualityFilterer: {self._varname} in [{self._minval:.3g}, {self._maxval:.3g}]>'

    def quality_flags(self, l2_file: str):
        with ncdf.Dataset(l2_file) as ds:
            values = readers.find_nc_var(ds, self._varname, return_type='array')
        return (values >= self._minval) & (values <= self._maxval)

    def sigma_out_of_bounds(self, l2_file: str):
        """Return a numpy vector that gives a normalized amount by which a given flagged sounding is "out-of-bounds"

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        sigmas
            A numpy float vector that describes how "out of bounds" a given sounding is. Good quality soundings
            will have a value of 0. For values that are above the upper limit (:math:`ul`), the value will be
            :math:`(v - ul)/(ul - ll)`. For values below the lower limit, the value will be the same except using
            :math:`ll` (the lower limit) in the numerator in place of :math:`ul`.
        """
        with ncdf.Dataset(l2_file) as ds:
            values = readers.find_nc_var(ds, self._varname, return_type='array')

        dv = self._maxval - self.minval
        sigmas = np.zeros(values.size)
        sigmas[values < self._minval] = (values[values < self._minval] - self._minval)/dv
        sigmas[values > self._maxval] = (values[values > self._maxval] - self._maxval)/dv
        return sigmas

    def reason(self, l2_file: str):
        flags = self.quality_flags(l2_file)
        reasons = np.full(flags.size, '', dtype=object)
        reasons[flags == 0] = f'{self._varname}_out_of_bounds'
        return reasons

    def get_plotting_df(self, l2_file: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        criteria
            A dataframe which will have a longitude and latitude as values in its row index, and the
            filter variable as the only column.
        """
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array'),
                self._varname: readers.find_nc_var(ds, self._varname, return_type='array')
            }

        return pd.DataFrame(data).set_index(['lon', 'lat'])



class MultipleQualityFilterer(AbstractQualityFilterer):
    """Quality filter that ANDs multiple filters.

    This class has some extra methods compared to the base quality filterers:

    * ``from_qa_df`` - create an instance from a dataframe representing a MUSES quality flagging file
    * ``add_filterer`` - add an additional filterer to the internal list.
    * ``remove_filterer`` - remove one or more of the existing filterers
    * ``reason_matrix`` - returns an :class:`xarray.DataArray` which defines which flags are marking
      each sounding as good or bad.

    Parameters
    ---------
    filterers
        A sequence of other :class:`AbstractQualityFilterer` child classes. The quality flags
        output by this filter will the the result of ANDing together the quality flags from
        each of these individual filters.
    """
    def __init__(self, filterers: Sequence[AbstractQualityFilterer]):
        self._filterers = list(filterers)

    def __repr__(self):
        inners = ', '.join(repr(f) for f in self._filterers)
        return f'<MultipleQualityFilterer: [{inners}]>'

    @classmethod
    def from_qa_df(cls, df: pd.DataFrame):
        """Create an instance of :class:`MultipleQualityFilterer` from a dataframe version of a MUSES quality file

        MUSES defines standard quality flags in the OSP directories under the "QualityFlags" subdirectory of a 
        strategy table. The :mod:`readers` module has two functions that can read those in as dataframes:
        :func:`~readers.read_quality_file` and :func:`~readers.read_quality_file_for_species`. Dataframes returned
        by those functions have columns: "Flag" (the name of the variable to flag on), "CutoffMin" and "CutoffMax"
        (the lower and upper limits), and "Use_For_Master" (whether that row should be used for the master quality flag).

        This function takes one of those dataframes, limits it to the rows with "Use_For_Master" set to 1, and creates
        an instance of this class that will reproduce the filtering on those variables.

        Parameters
        ----------
        df
            A dataframe representing a quality filtering strategy, as described above.

        Returns
        -------
        Self
            An instance of this class with one filter for each row of the input dataframe with "Use_For_Master" set to
            1.
        """
        filterers = cls._qa_df_to_filters(df)
        return cls(filterers)

    @staticmethod
    def _qa_df_to_filters(df: pd.DataFrame) -> Sequence[AbstractQualityFilterer]:
        df = df[df['Use_For_Master'] == 1]
        filterers = []
        for flag, row in df.iterrows():
            f = VariableQualityFilterer(varname=flag, minval=row['CutoffMin'], maxval=row['CutoffMax'])
            filterers.append(f)
        return filterers

    def add_filterer(self, filterer: AbstractQualityFilterer):
        """Add a new filter

        Parameters
        ----------
        filterer
            The new filter to add
        """
        self._filterers.append(filterer)

    def with_filterer(self, filterer: AbstractQualityFilterer):
        new = self.__class__(self._filterers)
        new.add_filterer(filterer)
        return new

    def remove_filterer(self, index_or_fxn: Union[int, Callable[[AbstractQualityFilterer], bool]]):
        """Remove one or more filters by index or checking if they meet certain criteria

        Parameters
        ----------
        index_or_fxn
            If this is an integer, it must be the index of the filter to remove. If a function,
            it must accept a filter as the only argument and return ``True`` if it should be removed.
        """
        if isinstance(index_or_fxn, int):
            if index_or_fxn < 0:
                index_or_fxn = len(self._filterers) + index_or_fxn
            self._filterers = [f for i, f in enumerate(self._filterers) if i != index_or_fxn]
        else:
            self._filterers = [f for f in self._filterers if not index_or_fxn(f)]

    @staticmethod
    def _get_ntgts(l2_file: str) -> int:
        """Internal method to get the number of targets in the L2 file
        """
        with ncdf.Dataset(l2_file) as ds:
            return ds.dimensions['Grid_Targets'].size

    def quality_flags(self, l2_file: str):
        """Return a numpy vector that is ``True`` for good soundings and ``False`` for bad ones.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        flags
            A numpy boolean vector; ``True`` = good quality, ``False`` = bad quality. This is ANDed across
            all the filters, so all filters must return ``True`` for a sounding to count as good.
        """
        ntgt = self._get_ntgts(l2_file)
        qf = np.ones(ntgt, dtype=bool)
        for filterer in self._filterers:
            qf &= filterer.quality_flags(l2_file)
        return qf

    def sigma_out_of_bounds(self, l2_file: str) -> np.ndarray:
        """Return a numpy vector that gives a normalized amount by which a given flagged sounding is "out-of-bounds"

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        sigmas
            A numpy float vector that describes how "out of bounds" a given sounding is. Good quality soundings
            will have a value of 0. For soundings flagged by any filter, the value will be the one from all filters
            that was furthest from 0.
        """
        sigmas = self._sigma_out_of_bounds_inner(l2_file)
        min_sigmas = np.nanmin(sigmas, axis=1)
        max_sigmas = np.nanmax(sigmas, axis=1)
        xx = np.abs(min_sigmas) > np.abs(max_sigmas)
        max_sigmas[xx] = min_sigmas
        return max_sigmas


    def _sigma_out_of_bounds_inner(self, l2_file: str):
        """Internal method that concatenates the out-of-bound vectors from each filter into a 2D array
        """
        ntgt = self._get_ntgts(l2_file)
        nfilt = len(self._filterers)
        sigmas = np.zeros([ntgt, nfilt], dtype=float)
        for ifilt, filterer in enumerate(self._filterers):
            sigmas[:,ifilt] = filterer.sigma_out_of_bounds(l2_file)
        return sigmas

    def reason(self, l2_file: str):
        """Return a string or vector of strings giving the biggest reason why a sounding was flagged

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        reasons
            A vector containing the reason each sounding was flagged. Soundings not flagged
            will have an empty string. Other soundings will have the string returned by the filter
            with the largest value from :meth:`sigma_out_of_bounds`. If you need to know all the
            reasons a sounding was flagged, see :meth:`reason_matrix`.
        """
        ntgt = self._get_ntgts(l2_file)
        nfilt = len(self._filterers)
        sigmas = self._sigma_out_of_bounds_inner(l2_file)
        indiv_reasons = np.full([ntgt, nfilt], '', dtype=object)
        for ifilt, filterer in enumerate(self._filterers):
            indiv_reasons[:, ifilt] = filterer.reason(l2_file)

        idx = np.nanargmax(np.abs(sigmas), axis=1)
        return indiv_reasons[(np.arange(ntgt, dtype=int), idx)]

    def reason_matrix(self, l2_file, sounding_id_var='SoundingID') -> xr.DataArray:
        """Return a matrix that indicates which soundings are flagged by which criteria.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        sounding_id_var
            The variable name in the file that contains the sounding IDs. It will be searched for
            using :func:`~readers.find_nc_var`, ignoring case. Set this to ``None`` to skip reading
            the sounding IDs and just use monotonic integers for that dimension of the output array.

        Returns
        -------
        matrix
            An :class:`xarray.DataArray` that will have dimensions "target" and "flag". The coordinates
            for "target" will be the sounding ID strings *unless* ``sounding_id_var`` was ``None``, in which
            case they will be just the row index. The coordinates for the "flag" dimension are the reason
            strings for each filter. If a filter returns >1 non-empty string in its vector of reasons,
            all the unique reasons for that filter get concatenated, joined with " + ".

            The values of the array will be 1 if the sounding is GOOD based on that criterion and 0 if it
            is BAD.
        """
        ntgt = self._get_ntgts(l2_file)
        nfilt = len(self._filterers)
        flags = np.zeros([ntgt, nfilt], dtype=bool)

        reasons = []
        for ifilt, filterer in enumerate(self._filterers):
            flags[:, ifilt] = filterer.quality_flags(l2_file)
            filt_reasons = ' + '.join(x for x in np.unique(filterer.reason(l2_file)) if len(x) > 0)
            reasons.append(filt_reasons)

        if sounding_id_var is None:
            sids = np.arange(ntgt, dtype=int)
        else:
            with ncdf.Dataset(l2_file) as ds:
                sids = readers.find_nc_var(ds, sounding_id_var, return_type='array')
                sids = readers.convert_sounding_ids(sids)

        return xr.DataArray(
            flags,
            dims=['target', 'flag'],
            coords=[sids, reasons]
        )

    def get_plotting_df(self, l2_file: str) -> pd.DataFrame:
        """
        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        criteria
            A dataframe which will have a longitude and latitude as values in its row index, and the
            filter variables from each internal filter as the columns.
        """
        def convert_index(idx):
            n = len(idx[0])
            return np.asarray([idx.get_level_values(i).to_numpy() for i in range(n)]).T


        dfs = [f.get_plotting_df(l2_file) for f in self._filterers]
        inds = [df.index for df in dfs]
        ind0 = convert_index(inds[0])

        if any(not np.allclose(ind0, convert_index(i)) for i in inds[1:]):
            raise NotImplementedError('Lat/lon indices are not within floating point error')
        dfs = [df.reset_index(drop=True) for df in dfs]
        full_df = pd.concat(dfs, axis=1)
        full_df.index = inds[0]
        return full_df

    def plot(self, l2_file: str, axs=None, **scatter_kws):
        """Make scatter plots of the variables filtered on.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        axs
            The sequence of matplotlib axes to plot into. This must be a 1D sequence with at least as many axes
            as filterers included in this instance. If too few are passed, a :class:`TypeError` is raised. If this
            argument isn't given, then axes are created in two columns. Each filter will get one set of axes to plot
            into. Handling filters that need >1 set of axes isn't yet implemented.

        scatter_kws
            Additional keyword arguments to pass to the :func:`~matplotlib.pyplot.scatter` plotting function. 

        Returns
        -------
        axs
            The axes or sequence of axes plotted into.
        """
        nplots = len(self._filterers)
        if axs is None:
            sp = jplt.Subplots(nplots, subplot_kw={'projection': ccrs.PlateCarree()})
            axs = [sp.next_subplot() for _ in range(nplots)]
        elif len(axs) < nplots:
            raise TypeError(f'Not enough sets of axes; needed {nplots}, got {len(axs)}')

        for ifilt, filterer in enumerate(self._filterers):
            filterer.plot(l2_file, axs=axs[ifilt], **scatter_kws)

        return axs


class StdCrisPanQualityFilter(MultipleQualityFilterer):
    """Quality filter for CrIS PAN.

    The default considers both the PAN step and the preceding H2O retrieval
    main quality flag. Additional filters can be passed during instantiation,
    if needed.
    """
    def __init__(self, *extra_filters):
        filters = [
            StdFlagQualityFilterer('Quality'),
            StdFlagQualityFilterer('Quality', alternate_step='H2O-1'),
        ]
        filters.extend(extra_filters)
        super().__init__(filters)


class ExtraPanSurfTFilter(AbstractQualityFilterer):
    def __init__(self, min_surf_temp: float = 265.0):
        self._min_surf_temp = min_surf_temp

    def _load_values(self, l2_file):
        with ncdf.Dataset(l2_file) as ds:
            surf_temp = readers.find_nc_var(ds, 'SurfaceTemperature', return_type='array')
        return surf_temp

    def quality_flags(self, l2_file: str):
        surf_temp = self._load_values(l2_file)
        return surf_temp > self._min_surf_temp

    def sigma_out_of_bounds(self, l2_file: str):
        flagged = ~self.quality_flags(l2_file)
        sigmas = np.zeros(flagged.shape)
        surf_temp = self._load_values(l2_file)
        sigmas[flagged] = (self._min_surf_temp - surf_temp) / self._min_surf_temp
        return sigmas

    def reason(self, l2_file: str) -> np.ndarray:
        flagged = ~self.quality_flags(l2_file)
        reasons = np.full(flagged.shape, '', dtype=object)
        reasons[flagged] = 'Surface temperature too small'
        return reasons

    def get_plotting_df(self, l2_file: str) -> pd.DataFrame:
        surf_temp = self._load_values(l2_file)
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array'),
                'delta_surf_temp': surf_temp - self._min_surf_temp,
            }
        return pd.DataFrame(data).set_index(['lon', 'lat'])


class ExtraPanCloudPFilter(AbstractQualityFilterer):
    def __init__(self, trop_pres_buf: float = 20.0):
        self._trop_pres_buf = trop_pres_buf

    def _load_values(self, l2_file):
        with ncdf.Dataset(l2_file) as ds:
            trop_pres = readers.find_nc_var(ds, 'TropopausePressure', return_type='array')
            cld_top_pres = readers.find_nc_var(ds, 'CloudTopPressure', return_type='array')

        return trop_pres, cld_top_pres

    def quality_flags(self, l2_file: str):
        trop_pres, cld_top_pres = self._load_values(l2_file)
        return trop_pres < (cld_top_pres + self._trop_pres_buf)

    def sigma_out_of_bounds(self, l2_file: str):
        good = self.quality_flags(l2_file) == 1
        trop_pres, cld_top_pres = self._load_values(l2_file)
        dp = (cld_top_pres - self._trop_pres_buf) - trop_pres
        sigma = dp / (cld_top_pres - self._trop_pres_buf)
        sigma[good] = 0
        return sigma

    def reason(self, l2_file: str) -> np.ndarray:
        flagged = self.quality_flags(l2_file)
        reasons = np.full(flagged.shape, '', dtype=object)
        reasons[flagged] = 'Cloud top pressure above tropopause'
        return reasons

    def get_plotting_df(self, l2_file: str) -> pd.DataFrame:
        trop_pres, cld_top_pres = self._load_values(l2_file)
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array'),
                'delta_trop_pres': trop_pres - (cld_top_pres + self._trop_pres_buf)
            }
        return pd.DataFrame(data).set_index(['lon', 'lat'])


class StdPanFilter(MultipleQualityFilterer):
    """A subclass of MultipleQualityFilterer that implements standard PAN filters

    This class overrides the standard __init__ method of :class:`MultipleQualityFilterer` to automatically build the list of filterers.

    Parameters
    ----------
    strategy_table
        The name of the strategy table to use the QA table from, e.g. "OSP-CrIS-v10-airs-windows".

    user
        The name of the user group to find the strategy table in. Unlike :func:`readers.read_quality_file_for_species`, this must be 
        specified. To use the operational tables, pass "ops".

    add_non_table_filters
        Set this to ``False`` to only include the QA filters based on the QA table, and not the bespoke ones defined in the code.
        Default is ``True``.

    add_airs_filter
        Use this to add an instance of :class:`AirsPanSlopeFilter` to the list of filterers. If this is ``None``, no instance is added.
        Otherwise it must be a string that is a valid first argument for ``AirsPanSlopeFilter`` (i.e. "deriv", "slope", or "slope_zero")
        or an actual instance of that class.
    """
    def __init__(self, strategy_table: str, user: str, add_non_table_filters: bool = True, add_airs_filter: Optional[str] = None, strat_table_dir=None):
        qa_df = readers.read_quality_file_for_species(strategy_table, species='PAN', user=user, strat_table_dir=strat_table_dir)
        df_filters = self._qa_df_to_filters(qa_df)
        super().__init__(df_filters)
        if add_non_table_filters:
            self.add_filterer(ExtraPanSurfTFilter())
            self.add_filterer(ExtraPanCloudPFilter())
        if isinstance(add_airs_filter, str):
            self.add_filterer(AirsPanSlopeFilter(add_airs_filter))
        elif isinstance(add_airs_filter, AirsPanSlopeFilter):
            self.add_filterer(add_airs_filter)
        elif add_airs_filter is not None:
            raise TypeError(f'Expected None, a string, or an instance of AirsPanSlopeFilter for `add_airs_filter`, instead got {add_airs_filter.__class__.__name__}')

class AirsPanSlopeFilter(AbstractFilterWithCache):
    """A filterer class that implements the experimental AIRS PAN filtering based on the change in slope of the radiance residuals

    Because these criteria are quite slow to calculate, this class will cache them once calculated. It will write them to a new file
    in the same directory as the L2 combined file. Note that this file may be rewritten if some of the values used in computing the
    metrics change. See the ``recalc`` parameter for details.

    .. note::
       This is for EXPERIMENTAL filtering; it should be replaced with metrics incorporated in the L2 files before release of the AIRS
       PAN product.

    Parameters
    ----------
    crit_mode
        Which criterion to use when flagging. Can be a string or variant of the :class:`CritMode` enumeration contained in this class.
        Values are:

        * "slope" - use the difference in absolute values of the final and initial residuals vs. frequency slopes above ``min_freq``.
        * "slope_zero" - similar to "slope", except that cases where the absolute values of both the initial and final slopes are
          less than ``near_zero_cutoff`` are set to a small negative value (half of the cutoff, to be precise). This way with an upper
          limit of 0, sounding where the slope was already near zero aren't filtered out if the slope increased very slightly.
        * "deriv" - use the absolute value of the final - initial XPAN difference divided by the "slope" criterion. The intent of this
          criterion is that, if the radiance residual slope did increase but the retrieved PAN didn't change much, then a reasonably
          small positive upper limit will still allow those cases through while filtering out truly bad cases.

    upper_limit
        Maximum value allowed for the criterion; values above this are flagged. Will be ``0`` by default when ``crit_mode`` is "slope"
        or "slope_zero" and ``25`` when ``crit_mode`` is "deriv".

    recalc
        When to regenerate the cache file. May be either a string or a variant of the :class:`RecalcMode` enum class inside this class.
        Options are:

        * "never" - do not create or regenerate the cache file under any circumstances. If there are reasons why it would need regenerated,
          a :class:`RuntimeError` is raised.
        * "if_missing" - only regenerate if the file is missing; if it is present but was created with different settings, raises a 
          :class:`RuntimeError`.
        * "if_needed" (default) - regenerates the cache file if it is missing or has settings that differ from those given to this instance.
        * "always" - regenerate the cache file no matter what.

    remove_outliers
        If true, then outlier residuals are removed before fitting the criterion slopes. A point must be an outlier in both the initial and
        final residuals to be removed, and outliers are defined as points exceed 5 median absolute deviations from the median.

    min_freq
        Only residuals for spectral points above this frequency (in cm-1) are fit for the radiance residual slopes.

    near_zero_cutoff
        For the "slope_zero" criterion only, see the ``crit_mode`` description for its meaning.

    rad_file_fxn
        A function that must accept two arguments (this class instance and the L2 combined file path) and return the path to the radiance
        file. If not given, the default behavior is to replace "L2-PAN-0" in the L2 filename with "Radiance-PAN" and look for a file with
        that name in the same directory as the L2 file.

    cache_file_fxn
        A function that must accept two arguments (this class instance and the L2 combined file path) and return the path to the cache file.
        If not given, the default behavior is to replace "L2" in the L2 filename with "QACache" and use a file with that name in the
        same directory as the L2 file.
    """
    class CritMode(Enum):
        SLOPE = 'slope'
        SLOPE_ZERO = 'slope_zero'
        DERIV = 'deriv'

    def __init__(self, crit_mode: CritMode, upper_limit: Optional[float] = None, recalc: RecalcMode = RecalcMode.IF_NEEDED,
                 remove_outliers: bool = False, min_freq: float = 790.0, near_zero_cutoff: float = 0.005, 
                 rad_file_fxn=None, cache_file_fxn=None):
        self._crit_mode = self.CritMode(crit_mode)
        self._upper_limit = self._default_ul(upper_limit, self._crit_mode)
        self._recalc = RecalcMode(recalc)
        self._remove_outliers = remove_outliers
        self._near_zero_cutoff = near_zero_cutoff
        self._min_freq = min_freq
        self._rad_file_fxn = rad_file_fxn
        self._cache_file_fxn = cache_file_fxn

    # ----------------------- #
    # Internal helper methods #
    # ----------------------- #

    @classmethod
    def _default_ul(cls, upper_limit, crit_mode):
        """Get the default upper limit for the given criterion mode, or return the user's upper limit
        """
        if upper_limit is not None:
            return upper_limit
        elif crit_mode == cls.CritMode.DERIV:
            # This value came from a test on the Pole Creek fire on 2022-09-21, it was a value that
            # excluded most of the expected bad soundings. However, that may have been using log mapping
            # instead of linear, so this may need updated.
            return 25.0
        else:
            # If in one of the radiance slope modes, the requirement is that the slope got closer to 0
            # (i.e. the criterion, |m_final| - |m_init|, should be negative). In the near zero case, borderline
            # cases have their criterion value deliberately set negative.
            return 0

    def _find_rad_file(self, l2_file: str) -> Path:
        """Find the radiance file; raise a FileNotFoundError if it is missing
        """
        if self._rad_file_fxn is None:
            rad_file = _find_rad_file_default(l2_file)
        else:
            rad_file = Path(self._rad_file_fxn(self, l2_file))

        if not rad_file.exists():
            raise FileNotFoundError(f'Cannot find radiance file, {rad_file}. If it does exist but with a different name, pass a custom rad_file_fxn to the class __init__ method')

        return rad_file

    def _find_cache_file(self, l2_file: str) -> Path:
        """Get the path to use for the cache file
        """
        if self._cache_file_fxn is None:
            l2_file = Path(l2_file)
            cache_fname = l2_file.name.replace('L2', 'QACache')
            cache_file = l2_file.parent / cache_fname
        else:
            cache_file = Path(self._cache_file_fxn(self, l2_file))

        return cache_file

    def get_pan_criterion(self, l2_file: str) -> np.ndarray:
        """Get the criterion values to flag on, regenerating the cache file if needed.
        """
        if self.is_recalc_needed(l2_file):
            print('Regenerating cache file...')
            criterion_data = self._calculate_criteria(l2_file)
            self._save_to_cache(l2_file, criterion_data)

        return self._load_from_cache(l2_file)

    def _is_cache_var_missing(self, cache_ds: ncdf.Dataset) -> bool:
        return 'PanRadSlopeCrit' not in cache_ds.groups['Characterization'].variables.keys()

    def _are_cache_attrs_wrong(self, cache_ds: ncdf.Dataset) -> bool:
        return not np.isclose(cache_ds.min_freq, self._min_freq) or not np.isclose(cache_ds.near_zero_cutoff, self._near_zero_cutoff)

    def _calculate_criteria(self, l2_file) -> dict:
        """Calculate the slope and derivative criteria from the L2 and radiance files.
        """
        rad_file = self._find_rad_file(l2_file)

        with ncdf.Dataset(l2_file) as ds:
            lat = readers.get_nc_var(ds, 'Latitude')
            lon = readers.get_nc_var(ds, 'Longitude')
            sid = ds['SoundingID'][:]

            # Also read the quantities we need for the derivative criterion at the same time
            # We use the second set of column variables because the integration limits most closely
            # match the lite file XPAN limits.
            pan_col_init = readers.get_nc_var(ds, 'Characterization/Column_Initial')[:,1]
            pan_col_final = readers.get_nc_var(ds, 'Retrieval/Column')[:,1]
            air_col = readers.get_nc_var(ds, 'Retrieval/Column_Air')[:,1]

        with ncdf.Dataset(rad_file) as ds:
            rad_sid = ds['SOUNDINGID'][:]
            # The radiance file SIDs had an extra unit dimension in my testing. Squeezing
            # both arrays will get rid of that.
            if not np.array_equal(np.squeeze(sid), np.squeeze(rad_sid)):
                raise RuntimeError(f'L2 file ({l2_file}) and radiance file ({rad_file}) have different sounding IDs')

            freq = readers.get_nc_var(ds, 'FREQUENCY')
            radiance_init_bt = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEFITINITIAL'))
            radiance_final_bt = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEFIT'))
            radiance_obs_bt = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEOBSERVED'))

            m_init = np.full_like(lat, np.nan)
            m_final = np.full_like(lat, np.nan)
            m_criterion = np.full_like(lat, np.nan)

        pbar = miscutils.ProgressBar(freq.shape[0], prefix='Calculating slopes')
        for i in range(freq.shape[0]):
            pbar.print_bar(i)
            xx = (freq[i] > self._min_freq) & np.isfinite(freq[i])
            if np.sum(xx) == 0:
                continue
            this_freq = freq[i][xx]
            this_delta_rad_init = radiance_obs_bt[i][xx] - radiance_init_bt[i][xx]
            this_delta_rad_final = radiance_obs_bt[i][xx] - radiance_final_bt[i][xx]

            if self._remove_outliers:
                yy = ~jstats.isoutlier(this_delta_rad_final, m=5) & ~jstats.isoutlier(this_delta_rad_init, m=5)
            else:
                yy = np.ones(this_freq.shape, dtype=bool)

            if (~yy).sum() > 1:
                print(f'\nWarning: {(~yy).sum()} spectra points removed from sounding {i}')
            m_init[i] = jstats.PolyFitModel(this_freq[yy], this_delta_rad_init[yy], model='robust').coeffs[1]
            m_final[i] = jstats.PolyFitModel(this_freq[yy], this_delta_rad_final[yy], model='robust').coeffs[1]
            m_criterion[i] = np.abs(m_final[i]) - np.abs(m_init[i]) 

        m_crit_alt = m_criterion.copy()
        xx = (np.abs(m_init) < self._near_zero_cutoff) & (np.abs(m_final) < self._near_zero_cutoff)
        m_crit_alt[xx] = -self._near_zero_cutoff / 2

        xpan_init = 1e9 * pan_col_init / air_col
        xpan_final = 1e9 * pan_col_final / air_col
        dxpan = xpan_final - xpan_init

        return {
            'SoundingID': sid,
            'Latitude': lat,
            'Longitude': lon,
            'PanRadSlopeInit': m_init,
            'PanRadSlopeFinal': m_final,
            'PanRadSlopeCrit': m_criterion,
            'PanRadSlopeCritAlt': m_crit_alt,
            'XPanInit': xpan_init,
            'XPanFinal': xpan_final,
            'dXPanCrit': np.abs(dxpan) /  m_criterion
        }

    def _load_from_cache(self, l2_file: str) -> np.ndarray:
        """Load the desired quality metric from the cache file. Cache file must exist.
        """
        cache_file = self._find_cache_file(l2_file)
        with ncdf.Dataset(cache_file) as ds:
            if self._crit_mode == self.CritMode.SLOPE_ZERO:
                return ds.groups['Characterization']['PanRadSlopeCritAlt'][:].filled(np.nan)
            elif self._crit_mode == self.CritMode.SLOPE:
                return ds.groups['Characterization']['PanRadSlopeCrit'][:].filled(np.nan)
            elif self._crit_mode == self.CritMode.DERIV:
                return ds.groups['Characterization']['dXPanCrit'][:].filled(np.nan)
            else:
                raise NotImplementedError(f'_load_criterion not implemented for crit_mode = {self._crit_mode}')

    def _save_to_cache(self, l2_file: str, crit_data: dict):
        """Create a netCDF file with the quality metrics written to it
        """
        nzco = self._near_zero_cutoff
        char_group_vars = ['PanRadSlopeInit', 'PanRadSlopeFinal', 'PanRadSlopeCrit', 'PanRadSlopeCritAlt', 'XPanInit', 'XPanFinal', 'dXPanCrit']
        atts = {
            'PanRadSlopeInit': dict(
                description=f'Slope of intensity vs. wavenumber in the PAN window initial radiance residuals vs. observed above {self._min_freq} cm-1',
                units='K cm',
                min_freq=self._min_freq
            ),
            'PanRadSlopeFinal': dict(
                description=f'Slope of intensity vs. wavenumber in the PAN window final radiance residuals vs. observed above {self._min_freq} cm-1',
                units='K cm',
                min_freq=self._min_freq
            ),
            'PanRadSlopeCrit': dict(
                description='The difference of |PanRadSlopeFinal| - |PanRadSlopeInit|',
                units='K cm',
            ),
            'PanRadSlopeCritAlt': dict(
                description=f'The same as PanRadSlopeCrit, but soundings whose PanRadSlopeInit and PanRadSlopeFinal absolute values are < {nzco} set to {-nzco/2} to pass soundings that did not change much in the retrieval.',
                units='K cm',
                min_freq=self._min_freq,
                near_zero_cutoff=self._near_zero_cutoff
            ),
            'XPanInit': dict(
                description='XPAN computed from the L2 file INITIAL column amounts, using the second set of columns',
                units='ppb',
            ),
            'XPanFinal': dict(
                description='XPAN computed from the L2 file FINAL column amounts, using the second set of columns',
                units='ppb',
            ),
            'dXPanCrit': dict(
                description='|XPanFinal - XPanInit| divided by PanRadSlopeCrit. The intent is to use the change in PAN to reflect the sensitivity of the retrieval to the change in radiance slope.',
                units='ppb (K cm)^-1'
            )
        }

        cache_file = self._find_cache_file(l2_file)
        nsnd = crit_data['Latitude'].size
        with ncdf.Dataset(cache_file, 'w') as ds:
            ds.createDimension('Grid_Targets', nsnd)

            nschar = crit_data["SoundingID"].shape[1]
            sid_dim_2 = f'string{nschar}'
            ds.createDimension(sid_dim_2, nschar)    
            ds.createGroup('Characterization')

            ds.min_freq = self._min_freq
            ds.near_zero_cutoff = self._near_zero_cutoff

            for varname, vardata in crit_data.items():
                grp = ds.groups['Characterization'] if varname in char_group_vars else ds
                dims = ['Grid_Targets', sid_dim_2] if varname == 'SoundingID' else ['Grid_Targets']
                var = grp.createVariable(varname, vardata.dtype, dims)
                var[:] = vardata
                var.setncatts(atts.get(varname, dict()))

    # ------------------------ #
    # Regular filterer methods #
    # ------------------------ #

    def quality_flags(self, l2_file: str):
        """Return a numpy vector that is ``True`` for good soundings and ``False`` for bad ones.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        flags
            A numpy boolean vector; ``True`` = good quality, ``False`` = bad quality. For this class,
            good quality soundings are defined as those with their quality metric below the upper limit.
        """
        crit_vals = self.get_pan_criterion(l2_file)
        return crit_vals <= self._upper_limit

    def sigma_out_of_bounds(self, l2_file: str):
        """Return a numpy vector that gives a normalized amount by which a given flagged sounding is "out-of-bounds"

        Good quality soundings will have 0. Values may be negative. Because the filtering for this class has no lower
        limit, the values are normalized by the upper limit alone for the "deriv" criteria mode, and the actual standard
        deviation of the slopes in the other modes.

        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        sigmas
            A numpy float vector that describes how "out of bounds" a given sounding is. Good quality soundings
            will have a value of 0.
        """
        crit_vals = self.get_pan_criterion(l2_file)
        crit_flags = self.quality_flags(l2_file)

        sigmas = np.zeros(crit_vals.shape)
        is_bad = crit_flags == 0
        if self._crit_mode == self.CritMode.DERIV:
            sd = self._upper_limit
        else:
            sd = np.nanstd(crit_vals, ddof=1)

        sigmas[is_bad] = crit_vals[is_bad] / sd
        return sigmas


    def reason(self, l2_file: str):
        crit_flags = self.quality_flags(l2_file)
        reasons = np.full(crit_flags.shape, '', dtype=object)
        if self._crit_mode == self.CritMode.SLOPE:
            cause = 'pan_m_resid_not_improved'
        elif self._crit_mode == self.CritMode.SLOPE_ZERO:
            cause = 'pan_m_resid_not_improved_not_zero'
        elif self._crit_mode == self.CritMode.DERIV:
            cause = 'dxpan_dmresid_too_large'
        else:
            raise NotImplementedError(f'No cause string defined for crit_mode = {self._crit_mode}')

        reasons[crit_flags == 0] = cause
        return reasons

    def get_plotting_df(self, l2_file: str):
        """
        Parameters
        ----------
        l2_file
            Path to the Level 2 MUSES file to filter.

        Returns
        -------
        criteria
            A dataframe which will have a longitude and latitude as values in its row index. The data columns
            will include "Criterion" (which will be the quality metric filtered on) and the initial/final slopes
            and XPAN values for more detailed plots.
        """
        data = {'Criterion': self.get_pan_criterion(l2_file)}
        cache_file = self._find_cache_file(l2_file)
        with ncdf.Dataset(cache_file) as ds:
            for key in ds.groups['Characterization'].variables.keys():
                if 'Crit' not in key:
                    # Assumes that only the variables that store one of the three criteria
                    # have the substring "Crit". If that is no longer true, this needs updated.
                    data[key] = ds.groups['Characterization'][key][:].filled(np.nan)

            data['lon'] = ds['Longitude'][:].filled(np.nan)
            data['lat'] = ds['Latitude'][:].filled(np.nan)

        return pd.DataFrame(data).set_index(['lon', 'lat'])



    def plot(self, l2_file: str, axs=None, **scatter_kws):
        if self._crit_mode == self.CritMode.SLOPE:
            label = '|m_final| - |m_init| (K cm)'            
        elif self._crit_mode == self.CritMode.SLOPE_ZERO:
            label = '|m_final| - |m_init|, adjusted (K cm)'
        elif self._crit_mode == self.CritMode.DERIV:
            label = '|XPAN_final - XPAN_init|/(|m_final| - |m_init|) (ppb K$^{-1}$ cm $^{-1}$)'
        else:
            raise NotImplementedError(f'No plot label defined for crit_mode = {self._crit_mode}')

        maxval = self._upper_limit
        minval = -10*self._upper_limit if not np.isclose(self._upper_limit, 0) else -self._near_zero_cutoff*10
        df = self.get_plotting_df(l2_file).loc[:, ['Criterion']].rename(columns={'Criterion': label})
        flags = self.quality_flags(l2_file)
        scatter_kws.setdefault('vmin', minval)
        scatter_kws.setdefault('vmax', maxval)
        return self._plot_inner(df, flags, axs=axs, **scatter_kws)


class TSlopeLowCloudPANFilterer(AbstractQualityFilterer):
    def __init__(self, upper_delta_bt_threshold=-4, lower_delta_bt_threshold=None, i1=12, i2=13, check_freqs=(781.534, 793.885), rad_file_fxn=None):
        self._upper_delta_bt_threshold = upper_delta_bt_threshold
        self._lower_delta_bt_threshold = lower_delta_bt_threshold
        self._i1 = i1
        self._i2 = i2
        self._check_freqs = check_freqs
        self._rad_file_fxn = rad_file_fxn

    def __repr__(self) -> str:
        if self._upper_delta_bt_threshold is not None and self._lower_delta_bt_threshold is not None:
            return f'<{self.__class__.__name__}(dBT ∉ [{self._lower_delta_bt_threshold}, {self._upper_delta_bt_threshold}])>'
        elif self._upper_delta_bt_threshold is not None:
            return f'<{self.__class__.__name__}(dBT > {self._upper_delta_bt_threshold})>'
        elif self._lower_delta_bt_threshold:
            return f'<{self.__class__.__name__}(dBT < {self._lower_delta_bt_threshold})>'
        else:
            return f'<{self.__class__.__name__}(dBT ∈ ℝ)>'

    def quality_flags(self, l2_file: str):
        # Return flags that are true when the difference is less negative than the threshold value
        delta_bt = self._read_delta_bt(l2_file)
        xx = np.zeros(delta_bt.shape, dtype=bool)
        if self._upper_delta_bt_threshold is not None:
            xx |= delta_bt > self._upper_delta_bt_threshold
        if self._lower_delta_bt_threshold is not None:
            xx |= delta_bt < self._lower_delta_bt_threshold
        return xx

    def sigma_out_of_bounds(self, l2_file: str):
        delta_bt = self._read_delta_bt(l2_file)
        is_bad = ~self.quality_flags(l2_file)
        sigma = np.zeros_like(delta_bt)

        # We'll use different calculations if both limits are set vs. just one. If both limits are set, we're failing soundings whose change in BT
        # is between the lower and upper limits, so use whichever limit is closer to calculate the out-of-bounds metric. Also, normalize so that 1
        # is exactly in the middle of the disallowed range.
        #
        # If only using one limit, then we're flagging all values below the upper limit or values above the lower limit. Since we don't have a span
        # here, normalize with the limit value, and do the subtraction so that the sigmas are positive for bad soundings.
        if self._upper_delta_bt_threshold is not None and self._lower_delta_bt_threshold is not None:
            span = (self._upper_delta_bt_threshold - self._lower_delta_bt_threshold) / 2
            sigma_lower = np.abs(delta_bt - self._lower_delta_bt_threshold) / span
            sigma_upper = np.abs(delta_bt - self._upper_delta_bt_threshold) / span

            sigma[is_bad] = np.minimum(sigma_lower, sigma_upper)
        elif self._upper_delta_bt_threshold is not None:
            sigma[is_bad] = (self._upper_delta_bt_threshold - delta_bt[is_bad]) / self._upper_delta_bt_threshold
        elif self._lower_delta_bt_threshold is not None:
            sigma[is_bad] = (delta_bt[is_bad] - self._lower_delta_bt_threshold) / self._lower_delta_bt_threshold

        return sigma

    def reason(self, l2_file: str):
        crit_flags = self.quality_flags(l2_file)
        reasons = np.full(crit_flags.shape, '', dtype=object)
        if self._check_freqs is not None:
            reasons[~crit_flags] = f'Different in observed BT between frequencies {self._check_freqs[0]} and {self._check_freqs[1]} was <= {self._upper_delta_bt_threshold}'
        else:
            reasons[~crit_flags] = f'Different in observed BT between indices {self._i1} and {self._i2} was <= {self._upper_delta_bt_threshold}'
        return reasons

    def get_plotting_df(self, l2_file: str):
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array')
            }

        data['delta_bt'] = self._read_delta_bt(l2_file)
        return pd.DataFrame(data).set_index(['lon', 'lat'])

    def plot(self, l2_file: str, axs=None, **scatter_kws):
        df = self.get_plotting_df(l2_file)
        flags = self.quality_flags(l2_file)
        vmin = scatter_kws.pop('vmin', df['delta_bt'].min())
        vmax = scatter_kws.pop('vmax', df['delta_bt'].max())
        if self._upper_delta_bt_threshold is not None and self._lower_delta_bt_threshold is not None:
            split_cmap, split_norm = _make_two_split_colormap(self._lower_delta_bt_threshold, self._upper_delta_bt_threshold, vmin=vmin, vmax=vmax)
        elif self._upper_delta_bt_threshold is not None:
            split_cmap, split_norm = _make_split_colormap(self._upper_delta_bt_threshold, vmin=vmin, vmax=vmax)
        elif self._lower_delta_bt_threshold is not None:
            split_cmap, split_norm = _make_split_colormap(self._upper_delta_bt_threshold, vmin=vmin, vmax=vmax)
        else:
            split_cmap, split_norm = None, None
        scatter_kws.setdefault('cmap', split_cmap)
        scatter_kws.setdefault('norm', split_norm)
        ax, cb = self._plot_inner(df, flags, axs=axs, **scatter_kws)
        ticks = cb.get_ticks()
        ticks = np.union1d(ticks, [self._upper_delta_bt_threshold])
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cb.set_ticks(ticks)
        return ax, cb


    def _read_delta_bt(self, l2_file: str):
        rad_file = self._find_rad_file(l2_file)
        with ncdf.Dataset(rad_file) as ds:
            freq = readers.get_nc_var(ds, 'FREQUENCY')
            bt = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEOBSERVED'))

        # First check that the frequencies match expected, if given
        if self._check_freqs is not None:
            chk_freq1, chk_freq2 = self._check_freqs
            self._test_freqs(freq[:,self._i1], chk_freq1, self._i1)
            self._test_freqs(freq[:,self._i2], chk_freq2, self._i2)


        return bt[:,self._i2] - bt[:,self._i1]

    @staticmethod
    def _test_freqs(file_freq, chk_freq, idx):
        if not np.all(np.isclose(file_freq, chk_freq) | np.isnan(file_freq)):
            warn(f'Some frequencies at index {idx} are not close to the expected frequency of {chk_freq}')

    def _find_rad_file(self, l2_file: str) -> Path:
        """Find the radiance file; raise a FileNotFoundError if it is missing
        """
        # TODO: in quick_plots, move these methods to a mixin class
        if self._rad_file_fxn is None:
            rad_file = _find_rad_file_default(l2_file)
        else:
            rad_file = Path(self._rad_file_fxn(self, l2_file))

        if not rad_file.exists():
            raise FileNotFoundError(f'Cannot find radiance file, {rad_file}. If it does exist but with a different name, pass a custom rad_file_fxn to the class __init__ method')

        return rad_file



class PCALowCloudPANFilterer(AbstractFilterWithCache):
    """Filter for clouds based on the magnitude of fitting FM-pantest residuals with previously derived principle components 

    Parameters
    ----------
    coeff_threshold
        Minimum allowed value for the PCA coefficient before a sounding is classified as a cloud.

    cloud_component_index
        Which principle component (by index) to use as the indicator of clouds. Must be less than ``use_n_components``.

    pca_file
        NetCDF file to read the previously determined PCAs from.

    use_n_components
        How many of the PCA in the above file to use in fitting. In the default case, only the first two show any structure;
        the rest are basically noise.

    rad_file_fxn
        A function that, given this instance and the L2 file path as a string, will produce a string that points to the FM-pantest
        radiance file.

    stride_for_testing
        How many soundings to skip for every one calculated. Useful when testing because calculating PCA coefficients for lots of
        soundings is slow.

    recalc_mode
        When to recalculate the PCA coefficients: always, never, if the cache file is missing, or if the cache file does not match (default).
    """
    def __init__(self, coeff_threshold: float = -10, cloud_component_index: int = 1, 
                 pca_file: str = '/tb/lt_ref17/laughner/notebook-data/camel/daily/20230202-cloud-filtering/west_coast_fire_pca.nc', 
                 use_n_components: int = 2, rad_file_fxn = None, stride_for_testing: int = 1, recalc_mode: RecalcMode = RecalcMode.IF_NEEDED):
        if cloud_component_index >= use_n_components:
            raise ValueError('cloud_component_index is greater than or equal to the number of components being fit')
        self._coeff_threshold = coeff_threshold
        self._cloud_com_index = cloud_component_index
        self._pca_file = pca_file
        self._use_n_com = use_n_components
        self._rad_file_fxn = rad_file_fxn
        self._stride_for_testing = stride_for_testing
        self._recalc = recalc_mode
        self._cache_file_fxn = None

    ## Filter methods ##
    def quality_flags(self, l2_file: str):
        # Return flags that are true when the difference is less negative than the threshold value
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        return self._coeff_to_flags(pca_coeffs, pca_codes)

    def sigma_out_of_bounds(self, l2_file: str):
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        xx_bad = self._coeff_to_flags(pca_coeffs, pca_codes)
        sigma = np.zeros_like(pca_coeffs)
        sigma[xx_bad] = (pca_coeffs[xx_bad] - self._coeff_threshold) / self._coeff_threshold
        return sigma

    def reason(self, l2_file: str):
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        reasons = np.full(pca_coeffs.shape, '', dtype=object)
        xx_bad = self._coeff_to_flags(pca_coeffs)

        reasons[xx_bad & pca_codes == 0] = 'PCA coefficient below allowed threshold'
        reasons[xx_bad & pca_codes == 1] = 'All-NaN spectrum'
        reasons[pca_codes == 2] = 'Fitting PCA coefficients failed (not counted as bad)'
        return reasons

    def get_plotting_df(self, l2_file: str):
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array')
            }

        data['pca_cloud_coeff'], data['fit_code'] = self._fit_pca(l2_file)
        return pd.DataFrame(data).set_index(['lon', 'lat'])

    def plot(self, l2_file: str, axs=None, **scatter_kws):
        df = self.get_plotting_df(l2_file)
        flags = self.quality_flags(l2_file)
        vmin = scatter_kws.pop('vmin', df['pca_cloud_coeff'].min())
        vmax = scatter_kws.pop('vmax', df['pca_cloud_coeff'].max())
        split_cmap, split_norm = _make_split_colormap(self._coeff_threshold, vmin=vmin, vmax=vmax)
        scatter_kws.setdefault('cmap', split_cmap)
        scatter_kws.setdefault('norm', split_norm)
        ax, cb = self._plot_inner(df, flags, axs=axs, **scatter_kws)
        ticks = cb.get_ticks()
        ticks = np.union1d(ticks, [self._coeff_threshold])
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cb.set_ticks(ticks)
        return ax, cb

    ## Cache methods ##
    def _find_cache_file(self, l2_file: str) -> Path:
        if self._cache_file_fxn is None:
            l2_file = Path(l2_file)
            cache_fname = l2_file.name.replace('L2', 'PCA_QACache')
            cache_file = l2_file.parent / cache_fname
        else:
            cache_file = Path(self._cache_file_fxn(self, l2_file))

        return cache_file


    def _is_cache_var_missing(self, cache_ds: ncdf.Dataset) -> bool:
        for var in ['pca_coeffs', 'pca_codes']:
            if var not in cache_ds.variables.keys():
                return True

        return False

    def _are_cache_attrs_wrong(self, cache_ds: ncdf.Dataset) -> bool:
        for k, v in self._cache_id_attrs.items():
            cache_val = getattr(cache_ds, k)
            if k == '_coeff_threshold' and not np.isclose(v, cache_val):
                return True
            elif v != cache_val:
                return True

        return False

    def _load_from_cache(self, l2_file: str):
        cache_file = self._find_cache_file(l2_file)
        with ncdf.Dataset(cache_file, 'r') as ds:
            return {
                'pca_coeffs': ds['pca_coeffs'][:],
                'pca_codes': ds['pca_codes'][:]
            }

    def _save_to_cache(self, l2_file: str, cache_data: dict):
        if 'pca_coeffs' not in cache_data or not np.ndim(cache_data['pca_coeffs']) == 1:
            raise ValueError('Expected a 1D array, "pca_coeffs", in the cache_data dictionary')

        attrs = {
            'pca_coeffs': {'description': 'Coefficients of the PCA that represents the presence of clouds'},
            'pca_codes': {
                'description': 'Numeric code indicating the quality of the PCA coefficient fit to the radiance',
                'code_meanings': '0 = good fit, 1 = radiance all NaNs/fill values, 2 = fit failed'
            }
        }
        cache_file = self._find_cache_file(l2_file)
        with ncdf.Dataset(cache_file, 'w') as ds:
            self._copy_id_vars(l2_file, ds)
            ds.setncatts(self._cache_id_attrs)
            for varname, arr in cache_data.items():
                var = ds.createVariable(varname, arr.dtype, ('Grid_Targets',))
                var[:] = arr
                var.setncatts(attrs.get(varname, dict()))

    @property
    def _cache_id_attrs(self) -> dict:
        # Don't need the threshold in the cache as it doesn't affect the computation of
        # anything in the cache
        id_attrs = ('_cloud_com_index', '_pca_file', '_use_n_com', '_stride_for_testing')
        return {a: getattr(self, a) for a in id_attrs}


    ## PCA Calculation methods ##

    def _coeff_to_flags(self, pca_coeffs, pca_codes):
        # Don't count cases where the fit failed as bad - that doesn't mean that the
        # retrieval shouldn't be believed, it just means that the PCAs we have couldn't
        # sufficiently accuratly reproduce the observed radiance
        xx_bad = (pca_coeffs < self._coeff_threshold) & (pca_codes == 0)
        xx_nans = pca_codes == 1
        return ~xx_bad | xx_nans

    @cache
    def _load_pca_info(self):
        """Load the precomputed PCAs that will be fit to the radiances
        """
        with ncdf.Dataset(self._pca_file) as ds:
            return {
                'components': ds['components'][:self._use_n_com],
                'freq': ds['frequency'][:],
                'mean': ds['mean'][:],
                'init_coeffs': ds['mean_coefficients'][:],
            }


    def _fit_pca(self, l2_file):
        if self.is_recalc_needed(l2_file):
            print('PCA cache file failed validation. Recomputing values')
            pca_coeffs, pca_codes = self._fit_pca_inner(l2_file)
            self._save_to_cache(l2_file, {'pca_coeffs': pca_coeffs, 'pca_codes': pca_codes})
            return pca_coeffs, pca_codes
        else:
            print('Loading PCA fits from cache')
            cache_data = self._load_from_cache(l2_file)
            return cache_data['pca_coeffs'], cache_data['pca_codes']

    def _fit_pca_inner(self, l2_file):
        pca_info = self._load_pca_info()
        rad_file = self._find_rad_file(l2_file)
        with ncdf.Dataset(rad_file) as ds:
            freq = readers.get_nc_var(ds, 'FREQUENCY')
            rad = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEOBSERVED'))


        nspec = np.shape(rad)[0]
        pca_coeffs = np.full(nspec, np.nan)
        pca_codes = np.full(nspec, -1)
        niter = len(range(0, nspec, self._stride_for_testing))
        pbar = miscutils.ProgressBar(niter, prefix='Fitting PCAs')
        for i in range(0, nspec, self._stride_for_testing):
            pbar.print_bar()

            this_freq = freq[i]
            this_rad = rad[i]

            if np.all(~np.isfinite(this_rad)):
                pca_codes[i] = 1
                continue

            # Avoid trying to fit NaNs or near-zero radiances - both will make it fail
            # (the PCAs were calculated for radiances with neither)
            xx = np.isfinite(this_rad) & (this_rad > 1)
            opt = self._fit_components_to_one_spectrum(
                clean_obs_rad=this_rad[xx],
                obs_freq=this_freq[xx],
                pca_components=pca_info['components'],
                pca_mean=pca_info['mean'],
                pca_freq=pca_info['freq'],
                init_coeffs=pca_info['init_coeffs']
            )

            if opt.success:
                pca_coeffs[i] = opt.x[self._cloud_com_index]
                pca_codes[i] = 0
            else:
                pca_codes[i] = 2

        return pca_coeffs, pca_codes


    @staticmethod
    def _fit_components_to_one_spectrum(clean_obs_rad, obs_freq, pca_components, pca_mean, pca_freq, init_coeffs, res=0.1):
        def resid(x, obs, mean, com):
            constr = mean + com * x[:, np.newaxis]
            return np.nansum((constr - obs)**2)

        # First we need to get the PCA components and the observed radiances on the same frequency grid
        # To avoid floating point issues, divide by some fraction and round to the lower int before comparing.
        # An intersect1d with proper tolerance would be better, but I don't see that in numpy.
        q_obs_freq = (obs_freq // res).astype(int)
        q_pca_freq = (pca_freq // res).astype(int)
        common_freq, obs_idx, pca_idx = np.intersect1d(q_obs_freq, q_pca_freq, assume_unique=True, return_indices=True)

        clean_obs_rad = clean_obs_rad[obs_idx]
        pca_components = pca_components[:, pca_idx]
        pca_mean = pca_mean[pca_idx]

        # Cut down the initial coefficients array to just the components we have
        ncom = np.shape(pca_components)[0]
        if np.size(init_coeffs) < ncom:
            raise ValueError('Insufficient initial coefficients')

        init_coeffs = init_coeffs[:ncom]

        opt = least_squares(resid, x0=init_coeffs, kwargs=dict(obs=clean_obs_rad, mean=pca_mean, com=pca_components))
        return opt

    def _find_rad_file(self, l2_file: str) -> Path:
        """Find the radiance file; raise a FileNotFoundError if it is missing
        """
        # TODO: in quality, move these methods to a mixin class
        if self._rad_file_fxn is None:
            rad_file = _find_rad_file_default(l2_file)
        else:
            rad_file = Path(self._rad_file_fxn(self, l2_file))

        if not rad_file.exists():
            raise FileNotFoundError(f'Cannot find radiance file, {rad_file}. If it does exist but with a different name, pass a custom rad_file_fxn to the class __init__ method')

        return rad_file


class PCALowCloudPANFiltererOld(AbstractQualityFilterer):
    """Filter for clouds based on the magnitude of fitting FM-pantest residuals with previously derived principle components 

    Parameters
    ----------
    coeff_threshold
        Minimum allowed value for the PCA coefficient before a sounding is classified as a cloud.

    cloud_component_index
        Which principle component (by index) to use as the indicator of clouds. Must be less than ``use_n_components``.

    pca_file
        NetCDF file to read the previously determined PCAs from.

    use_n_components
        How many of the PCA in the above file to use in fitting. In the default case, only the first two show any structure;
        the rest are basically noise.

    rad_file_fxn
        A function that, given this instance and the L2 file path as a string, will produce a string that points to the FM-pantest
        radiance file.

    stride_for_testing
        How many soundings to skip for every one calculated. Useful when testing because calculating PCA coefficients for lots of
        soundings is slow.

    cache_file
        Path to a pickle file to use for caching the PCA coefficients. If not given, then the coefficients will be recalculated if
        not cached in memory. If this is given but doesn't exist or doesn't match the parameters of the PCA fitting defined by this
        class, the coefficients are still recalculated but written to this file. Otherwise the coefficients are read from this file.

        .. note::
           This class does already use in-memory caching of the PCA coefficients, which is generally more reliable. Only use this
           option if you need to cache across Python instances, since otherwise a different class instance ID will skip the memory cache.

    require_cache
        If ``True``, a valid cache file *must* exist, otherwise an error is raise. This will never save a cache file.
    """
    def __init__(self, coeff_threshold: float = -10, cloud_component_index: int = 1, 
                 pca_file: str = '/tb/lt_ref17/laughner/notebook-data/camel/daily/20230202-cloud-filtering/west_coast_fire_pca.nc', 
                 use_n_components: int = 2, rad_file_fxn = None, stride_for_testing: int = 1, cache_file: Optional[str] = None, require_cache: bool = False):
        if cloud_component_index >= use_n_components:
            raise ValueError('cloud_component_index is greater than or equal to the number of components being fit')
        self._coeff_threshold = coeff_threshold
        self._cloud_com_index = cloud_component_index
        self._pca_file = pca_file
        self._use_n_com = use_n_components
        self._rad_file_fxn = rad_file_fxn
        self._stride_for_testing = stride_for_testing
        self._cache_file = cache_file
        self._require_cache = require_cache

    def quality_flags(self, l2_file: str):
        # Return flags that are true when the difference is less negative than the threshold value
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        return self._coeff_to_flags(pca_coeffs, pca_codes)

    def sigma_out_of_bounds(self, l2_file: str):
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        xx_bad = self._coeff_to_flags(pca_coeffs, pca_codes)
        sigma = np.zeros_like(pca_coeffs)
        sigma[xx_bad] = (pca_coeffs[xx_bad] - self._coeff_threshold) / self._coeff_threshold
        return sigma

    def reason(self, l2_file: str):
        pca_coeffs, pca_codes = self._fit_pca(l2_file)
        reasons = np.full(pca_coeffs.shape, '', dtype=object)
        xx_bad = self._coeff_to_flags(pca_coeffs)

        reasons[xx_bad & pca_codes == 0] = 'PCA coefficient below allowed threshold'
        reasons[xx_bad & pca_codes == 1] = 'All-NaN spectrum'
        reasons[pca_codes == 2] = 'Fitting PCA coefficients failed (not counted as bad)'
        return reasons

    def get_plotting_df(self, l2_file: str):
        with ncdf.Dataset(l2_file) as ds:
            data = {
                'lon': readers.find_nc_var(ds, 'longitude', return_type='array'),
                'lat': readers.find_nc_var(ds, 'latitude', return_type='array')
            }

        data['pca_cloud_coeff'], data['fit_code'] = self._fit_pca(l2_file)
        return pd.DataFrame(data).set_index(['lon', 'lat'])

    def plot(self, l2_file: str, axs=None, **scatter_kws):
        df = self.get_plotting_df(l2_file)
        flags = self.quality_flags(l2_file)
        vmin = scatter_kws.pop('vmin', df['pca_cloud_coeff'].min())
        vmax = scatter_kws.pop('vmax', df['pca_cloud_coeff'].max())
        split_cmap, split_norm = _make_split_colormap(self._coeff_threshold, vmin=vmin, vmax=vmax)
        scatter_kws.setdefault('cmap', split_cmap)
        scatter_kws.setdefault('norm', split_norm)
        ax, cb = self._plot_inner(df, flags, axs=axs, **scatter_kws)
        ticks = cb.get_ticks()
        ticks = np.union1d(ticks, [self._coeff_threshold])
        ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]
        cb.set_ticks(ticks)
        return ax, cb

    def _coeff_to_flags(self, pca_coeffs, pca_codes):
        # Don't count cases where the fit failed as bad - that doesn't mean that the
        # retrieval shouldn't be believed, it just means that the PCAs we have couldn't
        # sufficiently accuratly reproduce the observed radiance
        xx_bad = (pca_coeffs < self._coeff_threshold) & (pca_codes == 0)
        xx_nans = pca_codes == 1
        return ~xx_bad | xx_nans

    @cache
    def _load_pca_info(self):
        with ncdf.Dataset(self._pca_file) as ds:
            return {
                'components': ds['components'][:self._use_n_com],
                'freq': ds['frequency'][:],
                'mean': ds['mean'][:],
                'init_coeffs': ds['mean_coefficients'][:],
            }

    def _fit_pca(self, l2_file):
        # Not given a cache file, calculate the PCA (may use values cached in memory)
        if self._cache_file is None:
            return self._fit_pca_inner(l2_file)

        # If cache file given, but does not exist, then we need to recalculate anyway,
        # but this time save.
        if not Path(self._cache_file).exists():
            print(f'Cache file {self._cache_file} does not exist, must recompute')
            pca_coeffs, pca_codes = self._fit_pca_inner(l2_file)
            self._save_cache_file(pca_coeffs, pca_codes, l2_file)
            return pca_coeffs, pca_codes

        # Otherwise, open the cache and check to see if all the important variables are the
        # same. If so, we can use it; otherwise recalculate and overwrite.
        with open(self._cache_file, 'rb') as f:
            print(f'Trying to load from cache file {self._cache_file}...')
            prev_cache = pickle.load(f)

        failure_reasons = [] 
        if self._validate_cache_attrs(prev_cache, l2_file, failure_reasons=failure_reasons):
            print('Cache file validated. Using cached values')
            return prev_cache['pca_coeffs'], prev_cache['pca_codes']
        elif self._require_cache:
            causes = '\n* '.join(failure_reasons)
            raise RuntimeError(f'Cache file validation failed:\n* {causes}')
        else:
            print('PCA cache file failed validation. Recomputing values')
            pca_coeffs, pca_codes = self._fit_pca_inner(l2_file)
            self._save_cache_file(pca_coeffs, pca_codes, l2_file)
            return pca_coeffs, pca_codes

    def _save_cache_file(self, pca_coeffs, pca_codes, l2_file):
        if self._require_cache:
            print('Not saving cache due to require_cache=True (indicating we must use an existing cache file)')

        with open(self._cache_file, 'wb') as f:
            data = self._cache_id_attrs
            data.update({'pca_coeffs': pca_coeffs, 'pca_codes': pca_codes, 'l2_file': l2_file})
            pickle.dump(data, f)
            print(f'Saved PCA filter cache file {self._cache_file}')

    @property
    def _cache_id_attrs(self) -> dict:
        # Don't need the threshold in the cache as it doesn't affect the computation of
        # anything in the cache
        id_attrs = ('_cloud_com_index', '_pca_file', '_use_n_com', '_stride_for_testing')
        return {a: getattr(self, a) for a in id_attrs}

    def _validate_cache_attrs(self, cache_dict, l2_file, failure_reasons=None):
        if failure_reasons is None:
            failure_reasons = []

        if 'l2_file' not in cache_dict:
            failure_reasons.append('L2 file not listed in cache')
            return False
        elif Path(l2_file).resolve() != Path(cache_dict['l2_file']).resolve():
            failure_reasons.append(f'L2 file different than that in the cache: current {l2_file} vs. cached {cache_dict["l2_file"]}')
            return False

        status = True
        for k, v in self._cache_id_attrs.items():
            if k not in cache_dict:
                failure_reasons.append(f'{k} missing from cache')
                status = False

            if k == '_coeff_threshold' and not np.isclose(v, cache_dict[k]):
                failure_reasons.append(f'{k} is not within floating point tolerance: current {v} vs. cached {cache_dict[k]}')
                status = False
            elif v != cache_dict[k]:
                failure_reasons.append(f'{k} is not the same: current {v} vs. cached {cache_dict[k]}')
                status = False

        return status


    @cache
    def _fit_pca_inner(self, l2_file):
        pca_info = self._load_pca_info()
        rad_file = self._find_rad_file(l2_file)
        with ncdf.Dataset(rad_file) as ds:
            freq = readers.get_nc_var(ds, 'FREQUENCY')
            rad = conversions.bt(freq, readers.get_nc_var(ds, 'RADIANCEOBSERVED'))


        nspec = np.shape(rad)[0]
        pca_coeffs = np.full(nspec, np.nan)
        pca_codes = np.full(nspec, -1)
        niter = len(range(0, nspec, self._stride_for_testing))
        pbar = miscutils.ProgressBar(niter, prefix='Fitting PCAs')
        for i in range(0, nspec, self._stride_for_testing):
            pbar.print_bar()

            this_freq = freq[i]
            this_rad = rad[i]

            if np.all(~np.isfinite(this_rad)):
                pca_codes[i] = 1
                continue

            # Avoid trying to fit NaNs or near-zero radiances - both will make it fail
            # (the PCAs were calculated for radiances with neither)
            xx = np.isfinite(this_rad) & (this_rad > 1)
            opt = self._fit_components_to_one_spectrum(
                clean_obs_rad=this_rad[xx],
                obs_freq=this_freq[xx],
                pca_components=pca_info['components'],
                pca_mean=pca_info['mean'],
                pca_freq=pca_info['freq'],
                init_coeffs=pca_info['init_coeffs']
            )

            if opt.success:
                pca_coeffs[i] = opt.x[self._cloud_com_index]
                pca_codes[i] = 0
            else:
                pca_codes[i] = 2

        return pca_coeffs, pca_codes


    @staticmethod
    def _fit_components_to_one_spectrum(clean_obs_rad, obs_freq, pca_components, pca_mean, pca_freq, init_coeffs, res=0.1):
        def resid(x, obs, mean, com):
            constr = mean + com * x[:, np.newaxis]
            return np.nansum((constr - obs)**2)

        # First we need to get the PCA components and the observed radiances on the same frequency grid
        # To avoid floating point issues, divide by some fraction and round to the lower int before comparing.
        # An intersect1d with proper tolerance would be better, but I don't see that in numpy.
        q_obs_freq = (obs_freq // res).astype(int)
        q_pca_freq = (pca_freq // res).astype(int)
        common_freq, obs_idx, pca_idx = np.intersect1d(q_obs_freq, q_pca_freq, assume_unique=True, return_indices=True)

        clean_obs_rad = clean_obs_rad[obs_idx]
        pca_components = pca_components[:, pca_idx]
        pca_mean = pca_mean[pca_idx]

        # Cut down the initial coefficients array to just the components we have
        ncom = np.shape(pca_components)[0]
        if np.size(init_coeffs) < ncom:
            raise ValueError('Insufficient initial coefficients')

        init_coeffs = init_coeffs[:ncom]

        opt = least_squares(resid, x0=init_coeffs, kwargs=dict(obs=clean_obs_rad, mean=pca_mean, com=pca_components))
        return opt

    def _find_rad_file(self, l2_file: str) -> Path:
        """Find the radiance file; raise a FileNotFoundError if it is missing
        """
        # TODO: in quality, move these methods to a mixin class
        if self._rad_file_fxn is None:
            rad_file = _find_rad_file_default(l2_file)
        else:
            rad_file = Path(self._rad_file_fxn(self, l2_file))

        if not rad_file.exists():
            raise FileNotFoundError(f'Cannot find radiance file, {rad_file}. If it does exist but with a different name, pass a custom rad_file_fxn to the class __init__ method')

        return rad_file


class AbstractBiasCorr(ABC):
    @abstractmethod
    def apply_bias_corr(self, l2_file: str, values: np.ndarray) -> np.ndarray:
        """Given the L2 file from which the ``values`` were read, return a bias corrected version of ``values``.
        """
        pass



class CrisXpanBiasCorr(AbstractBiasCorr):
    # This is the amount to scale the _bias correction_ by to put it in the
    # given units. Since the bias correction assumes XPAN is in ppb, all scales
    # are relative to that.
    _units_to_scales = {'ppb': 1.0}

    def __init__(self, units='ppb'):
        self.units = units

    def apply_bias_corr(self, l2_file: str, values: np.ndarray) -> np.ndarray:
        h2o_file = self.find_h2o_file_for_bias_corr(l2_file)
        with ncdf.Dataset(h2o_file) as ds:
            h2o_column = readers.get_nc_var(ds, 'Retrieval/Column')[:,0]
        bc = (0.05 + 0.035e-23 * h2o_column) * self._units_to_scales[self.units]
        return values + bc

    @staticmethod
    def find_h2o_file_for_bias_corr(pan_file):
        pan_file = Path(pan_file)
        # Lite_Products_L2-PAN-0.nc -> Lite_Products_L2-H2O-1.nc or
        # Products_L2-PAN-0.nc -> Products_L2-H2O-1.nc so split on
        # the first dash. Want H2O-1 as the H2O step before the PAN
        # step.
        file_name_start = pan_file.name.split('-')[0]
        new_name = f'{file_name_start}-H2O-1.nc'
        h2o_file = pan_file.parent / new_name
        if not h2o_file.exists():
            raise FileNotFoundError(f'Could not find {new_name} alongside {pan_file}')
        else:
            return h2o_file


def _make_split_colormap(vcenter, vmin, vmax, bottom_cmap='inferno', top_cmap='cool'):
    from matplotlib import cm, colors
    all_colors = [
        cm.get_cmap(bottom_cmap)(np.linspace(0, 1, 256)),
        cm.get_cmap(top_cmap)(np.linspace(0, 1, 256))
    ]
    split_colors = colors.LinearSegmentedColormap.from_list('split_colors', np.vstack(all_colors))
    divnorm = colors.TwoSlopeNorm(vcenter=vcenter, vmin=vmin, vmax=vmax)
    return split_colors, divnorm


def _make_two_split_colormap(vsplit1, vsplit2, vmin, vmax, bottom_cmap='winter', middle_cmap='inferno', top_cmap='cool'):
    from matplotlib import cm, colors

    class TwoSplitNorm(colors.Normalize):
        def __init__(self, v1, v2, vmin, vmax, clip=False):
            self.v1 = v1
            self.v2 = v2
            super().__init__(vmin=vmin, vmax=vmax, clip=clip)

        def __call__(self, value, clip: Optional[bool] = None):
            y = np.interp(value, [self.vmin, self.v1, self.v2, self.vmax], [0, 0.33, 0.67, 1], left=-np.inf, right=np.inf)
            return np.ma.masked_array(y)

        def inverse(self, value):
            return np.interp(value, [0, 0.33, 0.67, 1], [self.vmin, self.v1, self.v2, self.vmax], left=-np.inf, right=np.inf)

    all_colors = [
        cm.get_cmap(bottom_cmap)(np.linspace(0, 1, 171)),
        cm.get_cmap(middle_cmap)(np.linspace(0, 1, 170)),
        cm.get_cmap(top_cmap)(np.linspace(0, 1, 171))
    ]
    split_colors = colors.LinearSegmentedColormap.from_list('split_colors', np.vstack(all_colors))
    divnorm = TwoSplitNorm(vsplit1, vsplit2, vmin, vmax)
    return split_colors, divnorm


def _find_rad_file_default(l2_file: str) -> Path:
    l2_file = Path(l2_file)
    if 'Lite' in l2_file.name:
        rad_fname = l2_file.name.replace('Lite_Products_L2', 'Products_Radiance').replace('-0', '')
    else:
        rad_fname = l2_file.name.replace('L2-PAN-0', 'Radiance-PAN')
    return l2_file.parent / rad_fname


def _find_related_step_default(l2_file: str, desired_step: str) -> Path:
    l2_file = Path(l2_file)
    # Assume we're dealing with file names like Lite_Products_L2-HDO-0.nc or Products_L2-CO-0.nc,
    # both of which have the first - where the prefix ends
    prefix, _ = l2_file.stem.split('-', 1)
    new_file = l2_file.parent / f'{prefix}-{desired_step}.nc'
    if not new_file.exists():
        raise ValueError(f'Could not file a file for step "{desired_step}" that is the same product and output as {l2_file}')
    return new_file
