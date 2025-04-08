"""Whereas the :mod:`plots` module has detailed plotting components,
the functions here are meant for quick look plots.
"""
from enum import Enum
from pathlib import Path
import re
from typing import Optional, Sequence, Tuple, Union, Callable
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import netCDF4 as ncdf
import numpy as np
from scipy.stats import binned_statistic_2d
import warnings

from jllutils import geography as jgeo
from jllutils import plots as jplt
from jllutils import miscutils
from . import readers, quality, conversions, images


def plot_xpan_from_combined(combined_file, column_index=1, day_flag=1,
                            map_features=tuple(),
                            bins=None, filterer: Optional[quality.AbstractQualityFilterer] = None,
                            ax=None, **plot_kws):
    """Plot PAN column average mole fraction from a L2 combined file

    This uses the second array column of PAN and air number densities to approximate the normal lite file XPAN
    without needing to do the slower integration.

    If ``bins`` is given, then this uses :func:`plot_binned_xgas_from_combined`. If not, then this uses
    :func:`plot_xgas_from_combined`. See those functions for information on the respective parameters.
    """
    if bins is None:
        plot_xgas_from_combined(
            combined_file=combined_file,
            column_index=column_index,
            day_flag=day_flag,
            map_features=map_features,
            filterer=filterer,
            ax=ax,
            **plot_kws
        )
    else:
        plot_binned_xgas_from_combined(
            combined_file=combined_file,
            bins=bins,
            column_index=column_index,
            day_flag=day_flag,
            map_features=map_features,
            filterer=filterer,
            ax=ax,
            **plot_kws
        )


def plot_xgas_from_combined(combined_file: str, column_index: Union[int, str], day_flag: Optional[int] = 1, 
                            map_features: Sequence[Union[str, Path, 'NaturalEarthFeature']] = tuple(), filterer: Optional[quality.AbstractQualityFilterer] = None,
                            ax=None, invert_colorbar: bool = False, **scatter_kws):
    """Plot a map of estimated column average mole fractions from a MUSES L2 combined file

    Parameters
    ----------
    combined_file
        Path the L2 combined file to read from.

    column_index
        One of several options:

        1. A string of the form "p\\d+", e.g. "p500" will find the level in `Species` closest to that pressure
           level.
        2. A string that corresponds to a 1D variable (any path acceptable to :func:`readers.get_nc_var`)
        3. An integer will be used as an array column index (i.e. ``i`` in ``arr[:, i]``) that selects the 
           desired target species and air partial columns from the "Retrieval/Column" and "Retrieval/Column_Air" 
           variables. The plotted Xgas values will be the ``Column`` divided by ``Column_Air``. Alternatively, 
           pass a string to read a 1D variable directly.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file).

    map_features
        Paths to :file:`.csv` files which contain map features that can be plotted by :func:`jllutils.geography.plot_shapes_from_csv`,
        or Cartopy Natural Earth features that can be added with ``ax.add_feature``. Each one will be added to the map axes.

    filterer
        A filterer object to remove poor quality soundings. Only soundings for which the ``quality_flags`` method on this object
        returns a ``1`` will be plotted.

    ax
        If given, axes to plot into. Must have been initialized with the PlateCarree projection from cartopy.

    invert_colorbar
        Set to ``True`` to invert the colorbar's y-axis, putting the max at the bottom. (Useful if plotting e.g. pressure.)

    scatter_kws
        Additional keywords are passed through to the :func:`~matplotlib.pyplot.scatter` plotting function. Some defaults are
        set: ``s = 6`` and ``vmin = 0``.
    """
    def variable(ds):
        return _get_plot_quantity(ds, column_index)

    plot_sounding_map_from_combined(
        combined_file=combined_file,
        variable=variable,
        day_flag=day_flag,
        map_features=map_features,
        filterer=filterer,
        ax=ax,
        invert_colorbar=invert_colorbar,
        **scatter_kws
    )


def plot_sounding_map_from_combined(combined_file: str, variable: Union[str, Callable], day_flag: Optional[int] = 1,
                                    map_features: Sequence[Union[str, Path, 'NaturalEarthFeature']] = tuple(),
                                    filterer: Optional[quality.AbstractQualityFilterer] = None, bias_corr: Optional[quality.AbstractBiasCorr] = None,
                                    ax=None, invert_colorbar: bool = False, cblabel: Optional[str] = None, dim2_idx: Optional[int] = None,
                                    cb_kws=dict(), **scatter_kws):
    """Plot a map of any variable in a combined file that has or can be transformed to a single value per sounding


    Parameters
    ----------
    combined_file
        Path the L2 combined file to read from.

    variable
        The name or path of the variable in the ``combined_file`` (the second argument to :func:`readers.get_nc_var`) *or*
        a function that, given the open combined filed netCDF dataset as its sole argument, returns the values to plot (as an array)
        and the color bar label to use (as a string).

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file).

    map_features
        Paths to :file:`.csv` files which contain map features that can be plotted by :func:`jllutils.geography.plot_shapes_from_csv`,
        or Cartopy Natural Earth features that can be added with ``ax.add_feature``. Each one will be added to the map axes.

    filterer
        A filterer object to remove poor quality soundings. Only soundings for which the ``quality_flags`` method on this object
        returns a ``1`` will be plotted.

    ax
        If given, axes to plot into. Must have been initialized with the PlateCarree projection from cartopy.

    invert_colorbar
        Set to ``True`` to invert the colorbar's y-axis, putting the max at the bottom. (Useful if plotting e.g. pressure.)

    cblabel
        Label to use for the colorbar; if not given, this is inferred from the variable name or taken as the second return
        argument of the `variable` function.

    dim2_idx
        Give this when trying to plot a variable that has two dimensions; the value plotted will be ``variable[:,dim2_idx]``.
        Trying to plot a 2D variable without this will raise a ``TypeError``.

    cb_kws
        A dictionary of keywords to give to the colorbar command. Note that ``ax`` and ``label`` are always given and cannot
        be overridden.

    scatter_kws
        Additional keywords are passed through to the :func:`~matplotlib.pyplot.scatter` plotting function. Some defaults are
        set: ``s = 6`` and ``vmin = 0``.
    """
    with ncdf.Dataset(combined_file) as ds:
        lon = readers.get_nc_var(ds, 'Longitude')
        lat = readers.get_nc_var(ds, 'Latitude')
        if isinstance(variable, str):
            values = readers.get_nc_var(ds, variable)
            cblabel_inferred = variable
        else:
            values, cblabel_inferred = variable(ds)

        if values.ndim == 2 and dim2_idx is not None:
            values = values[:,dim2_idx]
        elif values.ndim == 2:
            raise TypeError(f'Must provide ``dim2_idx`` keyword when trying to plot a 2D variable (values shape = {values.shape})')
        elif values.ndim != 1:
            raise NotImplementedError(f'Plotting {values.ndim}D variables is not supported')

        cblabel = cblabel or cblabel_inferred

        xx = np.ones(lon.shape, dtype=bool)
        if day_flag is not None:
            day_night_flag = readers.get_nc_var(ds, 'DayNightFlag')
            xx &= day_night_flag == day_flag

    if bias_corr is not None:
        values = bias_corr.apply_bias_corr(combined_file, values)

    if filterer is not None:
        qflags = filterer.quality_flags(combined_file)
        xx &= (qflags == 1)


    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})


    scatter_kws.setdefault('s', 6)
    scatter_kws.setdefault('vmin', 0)
    h = ax.scatter(lon[xx], lat[xx], c=values[xx], **scatter_kws)
    cb = plt.colorbar(h, ax=ax, label=cblabel, **cb_kws)
    if invert_colorbar:
        cb.ax.invert_yaxis()
    _add_features(map_features, ax)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False



def plot_binned_xgas_from_combined(combined_file: str, column_index: Union[int, str], bins = 10, day_flag: Optional[int] = 1, 
                                   map_features: Sequence[Union[str, Path, 'NaturalEarthFeature']] = tuple(), filterer: Optional[quality.AbstractQualityFilterer] = None,
                                   ax=None, invert_colorbar: bool = False, **pcolor_kws):
    """Plot a map of estimated column average mole fractions from a MUSES L2 combined file

    Parameters
    ----------
    combined_file
        Path the L2 combined file to read from.

    column_index
        One of several options:

        1. A string of the form "p\d+", e.g. "p500" will find the level in `Species` closest to that pressure
           level.
        2. A string that corresponds to a 1D variable (any path acceptable to :func:`readers.get_nc_var`)
        3. An integer will be used as an array column index (i.e. ``i`` in ``arr[:, i]``) that selects the 
           desired target species and air partial columns from the "Retrieval/Column" and "Retrieval/Column_Air" 
           variables. The plotted Xgas values will be the ``Column`` divided by ``Column_Air``. Alternatively, 
           pass a string to read a 1D variable directly.

    bins
        Input to :func:`scipy.stats.binned_stats_2d` on how to bin the data. See the same named parameter of
        that function for details.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file).

    map_features
        Paths to :file:`.csv` files which contain map features that can be plotted by :func:`jllutils.geography.plot_shapes_from_csv`,
        or Cartopy Natural Earth features that can be added with ``ax.add_feature``. Each one will be added to the map axes.

    filterer
        A filterer object to remove poor quality soundings. Only soundings for which the ``quality_flags`` method on this object
        returns a ``1`` will be plotted.

    ax
        If given, axes to plot into. Must have been initialized with the PlateCarree projection from cartopy.

    invert_colorbar
        Set to ``True`` to invert the colorbar's y-axis, putting the max at the bottom. (Useful if plotting e.g. pressure.)

    pcolor_kws
        Additional keywords are passed through to the :func:`~matplotlib.pyplot.pcolormesh` plotting function. Some defaults are
        set: ``vmin = 0``.

    Returns
    -------
    means
        The plotted 2D array of binned mean values.

    lonedge
        The 1D array of longitude edges used in plotting.

    latedge
        The 1D array of latitude edges used in plotting.
    """

    with ncdf.Dataset(combined_file) as ds:
        lon = readers.get_nc_var(ds, 'Longitude')
        lat = readers.get_nc_var(ds, 'Latitude')
        xgas, cblabel = _get_plot_quantity(ds, column_index)

        xx = np.ones(lon.shape, dtype=bool)
        if day_flag is not None:
            day_night_flag = readers.get_nc_var(ds, 'DayNightFlag')
            xx &= day_night_flag == day_flag

    if filterer is not None:
        qflags = filterer.quality_flags(combined_file)
        xx &= (qflags == 1)

    lon = lon[xx]
    lat = lat[xx]
    xgas = xgas[xx]

    means, lonedge, latedge, _ = binned_statistic_2d(lon, lat, xgas, bins=bins)
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    pcolor_kws.setdefault('vmin', 0)
    h = ax.pcolormesh(lonedge, latedge, means.T, **pcolor_kws)
    cb = plt.colorbar(h, ax=ax, label=cblabel)
    if invert_colorbar:
        cb.ax.invert_yaxis()
    _add_features(map_features, ax)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False

    return means, lonedge, latedge


def _get_plot_quantity(ds: ncdf.Dataset, column_index: int):
    # Try to get the gas from the file name
    combined_file = ds.filepath()
    m = re.match(r'Products_L2-(\w+)', Path(combined_file).stem)
    if m is not None:
        gas_label = f'{m.group(1)} VMR'
    else:
        gas_label = 'VMR'

    with ncdf.Dataset(combined_file) as ds:
        if isinstance(column_index, str) and re.match(r'p\d+', column_index):
            pres = readers.get_nc_var(ds, 'Pressure')
            with warnings.catch_warnings():
                # There will almost always be an empty slice in these, suppress the "mean of empty slice" warning
                warnings.simplefilter('ignore')
                pres_mean = np.nanmean(pres, axis=0)
            tgt_pres = float(column_index[1:])
            idx = np.nanargmin(np.abs(pres_mean - tgt_pres))
            xgas = readers.get_nc_var(ds, 'Species')[:, idx]
            cblabel = f'{gas_label} near {tgt_pres} hPa'
        elif isinstance(column_index, str):
            xgas = readers.get_nc_var(ds, column_index)
            cblabel = column_index
        else:
            column = readers.get_nc_var(ds, 'Retrieval/Column')[:, column_index]
            column_air = readers.get_nc_var(ds, 'Retrieval/Column_Air')[:, column_index]
            xgas = column / column_air
            pmax = np.nanmean(readers.get_nc_var(ds, 'Characterization/Column_PressureMax')[:, column_index])
            pmin = np.nanmean(readers.get_nc_var(ds, 'Characterization/Column_PressureMin')[:, column_index])
            cblabel = f'Mean {gas_label} ({pmax:.1f} to {pmin:.1f} hPa)'

    return xgas, cblabel


def plot_vmr_at_pres_from_combined(combined_file: str, tgt_pres: float, day_flag: Optional[int] = 1,
                                   map_features: Sequence[Union[str,Path,'NaturalEarthFeature']] = tuple(), filterer: Optional[quality.AbstractQualityFilterer] = None,
                                   ax=None, **scatter_kws):
    """Plot a map of the VMR nearest a given pressure level from a MUSES L2 combined file

    Parameters
    ----------
    combined_file
        Path the L2 combined file to read from.

    tgt_pres
        Desired pressure level in hPa. This function will find the level with the mean pressure closes to this one
        and select that to plot. The mean pressure and standard deviation (to one decimal place) will be included
        in the colorbar label.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file).

    map_features
        Paths to :file:`.csv` files which contain map features that can be plotted by :func:`jllutils.geography.plot_shapes_from_csv`,
        or Cartopy Natural Earth features that can be added with ``ax.add_feature``. Each one will be added to the map axes.

    filterer
        A filterer object to remove poor quality soundings. Only soundings for which the ``quality_flags`` method on this object
        returns a ``1`` will be plotted.

    ax
        If given, axes to plot into. Must have been initialized with the PlateCarree projection from cartopy.

    scatter_kws
        Additional keywords are passed through to the :func:`~matplotlib.pyplot.scatter` plotting function. Some defaults are
        set: ``s = 6`` and ``vmin = 0``.
    """
    warnings.warn('This function deprecated, use plot_xgas_from_combined with `column_index = f"p{tgt_pres}"`')
    with ncdf.Dataset(combined_file) as ds:
        lon = readers.get_nc_var(ds, 'Longitude')
        lat = readers.get_nc_var(ds, 'Latitude')
        pres = readers.get_nc_var(ds, 'Pressure')
        with warnings.catch_warnings():
            # There will almost always be an empty slice in these, suppress the "mean of empty slice" warning
            warnings.simplefilter('ignore')
            pres_mean = np.nanmean(pres, axis=0)
            pres_std = np.nanstd(pres, axis=0, ddof=1)
        idx = np.nanargmin(np.abs(pres_mean - tgt_pres))
        vmrs = readers.get_nc_var(ds, 'Species')[:, idx]



        xx = np.ones(lon.shape, dtype=bool)
        if day_flag is not None:
            day_night_flag = readers.get_nc_var(ds, 'DayNightFlag')
            xx &= day_night_flag == day_flag

    if filterer is not None:
        qflags = filterer.quality_flags(combined_file)
        xx &= (qflags == 1)


    if ax is None:
        _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})


    scatter_kws.setdefault('s', 6)
    scatter_kws.setdefault('vmin', 0)
    h = ax.scatter(lon[xx], lat[xx], c=vmrs[xx], **scatter_kws)
    plt.colorbar(h, ax=ax, label=f'VMR at {pres_mean[idx]:.1f} $\\pm$ {pres_std[idx]:.1f} hPa')
    _add_features(map_features, ax)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False


def plot_xgas_and_filters(combined_file: str, column_index: int, filterer: quality.MultipleQualityFilterer,
                          day_flag: Optional[int] = 1, map_features: Sequence[Union[str,Path,'NaturalEarthFeature']] = tuple(), 
                          axs=None, xgas_kws=None, filt_kws=None, scatter_kws=None):
    """Plot an Xgas value from a level 2 combined file along with quantities used to filter it.

    Parameters
    ----------
    combined_file
        Path the L2 combined file to read from.

    column_index
        Array column index (i.e. ``i`` in ``arr[:, i]``) that selects the desired target species and air
        partial columns from the "Retrieval/Column" and "Retrieval/Column_Air" variables. The plotted Xgas
        values will be the ``Column`` divided by ``Column_Air``.

    filterer
        A quality filterer that will both be used to filter the Xgas values plotted and to plot the individual variables
        or metrics used in the filtering. Note that this must be a :class:`quality.MultipleQualityFilterer`, if only a
        single filter is needed, it must be wrapped in one of these.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file).

    map_features
        Paths to :file:`.csv` files which contain map features that can be plotted by :func:`jllutils.geography.plot_shapes_from_csv`,
        or Cartopy Natural Earth features that can be added with ``ax.add_feature``. Each one will be added to the map axes.

    axs
        If given, a list of axes to plot into. The first one will have the Xgas mole fractions plotted into it, the rest will get each of the 
        filtering metrics into. Each axes must have been initialized with the PlateCarree projection and the list must have at least as many 
        elements as the number of filterers in the :class:`quality.MultipleQualityFilterer` plus one for the Xgas values.

    xgas_kws
        Additional keywords to pass to the call to :func:`matplotlib.pyplot.scatter` when plotting the Xgas values. Defaults are set by
        :func:`plot_xgas_from_combined`.

    filt_kws
        Same as ``xgas_kws``, but for the quality metric plots. Defaults are mostly those defined in the individual filterer's plot methods,
        though the colormap will be set to a "coolwarm" amp with under points as "black" and overs as "lime", to make the metric plots visually
        distinct from the Xgas plot.

    scatter_kws
        Common keywords added to both ``xgas_kws`` and ``filt_kws``. Values in those dictionaries take precedence. Note that unlike other quick
        plotting functions, this is actually a dictionary, rather than just extra keywords.
    """
    nplots = 1 + len(filterer._filterers)
    if axs is None:
        sp = jplt.Subplots(nplots, subplot_kw={'projection': ccrs.PlateCarree()})
        ax_xgas = sp.next_subplot()
        axs_filt = [sp.next_subplot() for _ in range(nplots-1)]
    else:
        ax_xgas = axs[0]
        axs_filt = axs[1:]

    if scatter_kws is None:
        scatter_kws = dict()

    tmp = scatter_kws.copy()
    if xgas_kws:
        tmp.update(xgas_kws)
    xgas_kws = tmp

    tmp = scatter_kws.copy()
    if filt_kws:
        tmp.update(filt_kws)
    filt_kws = tmp
    filt_kws.setdefault('cmap', jplt.colormap_out_of_bounds('coolwarm', under='black', over='lime'))

    plot_xgas_from_combined(
        combined_file=combined_file,
        column_index=column_index,
        day_flag=day_flag,
        map_features=map_features,
        filterer=filterer,
        ax=ax_xgas,
        **xgas_kws
    )

    filterer.plot(combined_file, axs_filt, **filt_kws)
    for ax in axs_filt:
        _add_features(map_features, ax)
        gl = ax.gridlines(draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False


# -------------- #
# Radiance plots #
# -------------- #

class RadiancePlotType(Enum):
    """Defines the types of radiance plots available

    Options are:

    * 'spaghetti' - shows every individual radiance/residual vector as thin lines and the mean
      or other summary vector as a thicker line
    * 'quantile' - shows various quantiles of the radiances/residuals
    * 'hexbin' - shows the summary radiance/residual vector overplotted on a hexbin plot of
      the radiance vs. frequency density
    """
    SPAGHETTI = 'spaghetti'
    QUANTILE = 'quantile'
    HEXBIN = 'hexbin'


class FmRadianceType(Enum):
    """Defines what quantity to use for radiance/residuals in radiance plots

    Options are:

    * 'init' - shows the initial modeled radiances
    * 'final' - shows the final modeled radiances
    * 'obs' - shows the observed radiances
    * 'init-resid' - shows the residuals, initial - obs
    * 'final-resid' - show the residuals, final - obs 
    """
    INIT = 'init'
    FINAL = 'final'
    OBS = 'obs'
    INIT_RESID = 'init-resid'
    FINAL_RESID = 'final-resid'

    @classmethod
    def all_types(cls):
        return [getattr(cls, a) for a in dir(cls) if not a.startswith('__')]


def plot_radiance_residuals(combined_file: str, plot_type: Union[str, RadiancePlotType], rad_type: Union[str, FmRadianceType] = 'final-resid',
                            day_flag: Optional[int] = 1, dnf_file: Optional[str] = None, box: Optional[Tuple[float, float, float, float]] = None,
                            filterer: Optional[quality.AbstractQualityFilterer] = None, filter_near_zero_obs: bool = False,
                            ax=None, **plot_kws):
    """Plot radiance or residuals from a combined Radiance file

    Parameters
    ----------
    combined_file
        Path to the combined radiance file to read from

    plot_type
        Which type of radiance plot to make; can be a string convertible to a :class:`RadiancePlotType` or an instance of the latter.

    rad_type
        What quantity to use for the radiances or residuals; can be a string convertible to a :class:`FmRadianceType` or an instance of the latter.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file). Requires ``dnf_file`` if this is not ``None``.

    dnf_file
        Path to an L2 file that has day/night flags for the soundings stored in the radiance ``combined_file``. Note that no checking of
        sounding IDs is done currently.

    box
        If given, a 4-element sequence giving ``lonmin, lonmax, latmin, latmax`` to plot radiances/residuals from in the combined file.

    filterer
        A quality filterer that will both be used to filter the Xgas values plotted and to plot the individual variables
        or metrics used in the filtering. Note that this must be a :class:`quality.MultipleQualityFilterer`, if only a
        single filter is needed, it must be wrapped in one of these.

    filter_near_zero_obs
        Set to ``True`` to change any observed radiances less than 1 K brightness temperature to NaNs.

    ax
        Axes to plot into.

    plot_kws
        Additional keywords that can change the appearance of the plot generated. Available options depend on the value of ``plot_type``:

        For ``plot_type = "spaghetti"``:

        * ``summary_op`` is the operation to use to generate the summary vectors of radiances/residuals and frequencies.
          It must accept a numpy array as the first argument and ``axis=0`` as a keyword argument to apply it over the
          soundings. Default is :func:`numpy.nanmean`
        * ``indiv_lw`` is the linewidth for the individual radiances/residuals. Default is 0.5.
        * ``indiv_color`` is the color for the individual radiances/residuals. Default is "gray".
        * ``summary_lw`` is the linewidth for the summary radiances/residuals. Default is 2.
        * ``summary_color`` is the color for the individual radiances/residuals. Default is "black".

        For ``plot_type = "hexbin"``, ``summary_op``, ``summary_lw``, and ``summary_color`` are the same as for ``plot_type = "spaghetti"``.
        Any additional keywords are passed through to :func:`~matplotlib.pyplot.hexbin`, with defaults ``mincnt = 1`` and ``cmap = YlGn``.

        For ``plot_type = "quantile"``:

        * ``quantiles`` is a list of the quantiles (in the range 0 to 1) to plot. Default is ``[0.25, 0.5, 0.75]``.
        * Other keywords accepted by `plot` may be given as well, though is is advised to avoid 'color'.
    """
    plot_type = RadiancePlotType(plot_type)
    rad_type = FmRadianceType(rad_type)
    with ncdf.Dataset(combined_file) as ds:
        freq = readers.get_nc_var(ds, 'FREQUENCY')
        if rad_type in (FmRadianceType.INIT_RESID, FmRadianceType.FINAL_RESID, FmRadianceType.OBS):
            obs_radiances = readers.get_nc_var(ds, 'RADIANCEOBSERVED')
        else:
            obs_radiances = None

        if rad_type in (FmRadianceType.INIT, FmRadianceType.INIT_RESID):
            fit_radiances = readers.get_nc_var(ds, 'RADIANCEFITINITIAL')
        elif rad_type in (FmRadianceType.FINAL, FmRadianceType.FINAL_RESID):
            fit_radiances = readers.get_nc_var(ds, 'RADIANCEFIT')
        elif rad_type == FmRadianceType.OBS:
            fit_radiances = None
        else:
            raise NotImplementedError(f'{rad_type=}')


        with warnings.catch_warnings():
            # We get lots of divide-by-zero warnings which are annoying
            warnings.simplefilter('ignore')
            if fit_radiances is not None:
                fit_radiances = conversions.bt(freq, fit_radiances)
            if obs_radiances is not None:
                obs_radiances = conversions.bt(freq, obs_radiances)
                if filter_near_zero_obs:
                    obs_radiances[obs_radiances < 1] = np.nan

        if rad_type == FmRadianceType.INIT:
            resid = fit_radiances
            label = 'Initial fit BT (K)'
        elif rad_type == FmRadianceType.FINAL:
            resid = fit_radiances
            label = 'Final fit BT (K)'
        elif rad_type == FmRadianceType.OBS:
            resid = obs_radiances
            label = 'Observed BT (K)'
        elif rad_type == FmRadianceType.INIT_RESID:
            resid = fit_radiances - obs_radiances
            label = 'Init. residual BT (init - obs, K)'
        elif rad_type == FmRadianceType.FINAL_RESID:
            resid = fit_radiances - obs_radiances
            label = 'Final residual BT (final - obs, K)'
        else:
            raise NotImplementedError(f'{rad_type=}')

        xx_keep = np.ones(freq.shape[0], dtype=bool)
        if box is not None:
            lon = readers.get_nc_var(ds, 'LONGITUDE')
            lat = readers.get_nc_var(ds, 'LATITUDE')
            x1, x2, y1, y2 = box
            xx_keep &= (lon >= x1) & (lon <= x2) & (lat >= y1) & (lat <= y2)

    if day_flag is not None and dnf_file is None:
        raise TypeError('dnf_file must be given if day_flag is not None')
    elif day_flag is not None:
        with ncdf.Dataset(dnf_file) as ds:
            day_night_flag = readers.get_nc_var(ds, 'DayNightFlag')
            xx_keep &= day_night_flag == day_flag

    if filterer is not None:
        try:
            qflags = filterer.quality_flags(combined_file)
        except KeyError:
            if dnf_file is not None:
                qflags = filterer.quality_flags(dnf_file)
            else:
                raise
        xx_keep &= (qflags == 1)


    if ax is None:
        _, ax = plt.subplots()

    if plot_type == RadiancePlotType.SPAGHETTI:
        return _radiance_spaghetti_plot(freq[xx_keep], resid[xx_keep], label, ax, **plot_kws)
    elif plot_type == RadiancePlotType.HEXBIN:
        return _radiance_hexbin_plot(freq[xx_keep], resid[xx_keep], label, ax, **plot_kws)
    elif plot_type == RadiancePlotType.QUANTILE:
        return _radiances_quantile_plot(freq[xx_keep], resid[xx_keep], label, ax, **plot_kws)
    else:
        raise NotImplementedError(f'{plot_type=}')

def _radiance_spaghetti_plot(freq, resid, label, ax, summary_op=np.nanmean, indiv_lw=0.5, indiv_color='gray', summary_lw=2, summary_color='black'):
    ax.plot(freq.T, resid.T, color=indiv_color, lw=indiv_lw)
    with warnings.catch_warnings():
        # Ignore mean of empty slice warnings
        warnings.simplefilter('ignore')
        sum_freq = summary_op(freq, axis=0)
        sum_resid = summary_op(resid, axis=0)
    ax.plot(sum_freq, sum_resid, color=summary_color, lw=summary_lw)
    ax.set_xlabel('Frequency (cm$^{-1}$)')
    ax.set_ylabel(label)


def _radiance_hexbin_plot(freq, resid, label, ax, summary_op=np.nanmean, summary_lw=2, summary_color='black', **hexbin_kw):
    hexbin_kw.setdefault('mincnt', 1)
    hexbin_kw.setdefault('cmap', 'YlGn')
    h = ax.hexbin(freq.flatten(), resid.flatten(), **hexbin_kw)
    plt.colorbar(h, ax=ax, label='N')
    with warnings.catch_warnings():
        # Ignore mean of empty slice warnings
        warnings.simplefilter('ignore')
        sum_freq = summary_op(freq, axis=0)
        sum_resid = summary_op(resid, axis=0)
    ax.plot(sum_freq, sum_resid, color=summary_color, lw=summary_lw)
    ax.set_xlabel('Frequency (cm$^{-1}$)')
    ax.set_ylabel(label)


def _radiances_quantile_plot(freq, resid, label, ax, quantiles=[0.25, 0.5, 0.75], **style):
    with warnings.catch_warnings():
        # Ignore mean of empty slice warnings
        warnings.simplefilter('ignore')
        freq = np.nanmean(freq, axis=0)
        resid_qs = np.quantile(resid, quantiles, axis=0)

    for q, r in zip(quantiles, resid_qs):
        ax.plot(freq, r, label=f'{q} quantile', **style)
    ax.set_xlabel('Frequency (cm$^{-1}$)')
    ax.set_ylabel(label)
    ax.legend()


def plot_radiances_in_boxes(combined_file: str, plot_type: Union[str, RadiancePlotType], boxes: Sequence[Tuple[float, float, float, float]],
                            radiance_types = FmRadianceType.all_types(), image: images.ImageProvider = images.StockImageProvider(),
                            day_flag: Optional[int] = 1, dnf_file: Optional[str] = None, 
                            filterer: Optional[quality.AbstractQualityFilterer] = None, filter_near_zero_obs: bool = False,
                            map_extent='global', **rad_plot_kws):
    """Plot a map followed by radiance plots from various boxes in the map

    The result will be a map followed by a grid of radiance/residual plots. Each row of these radiance/residual plots will be
    for one box, each column will be for one radiance/residual quantity.

    Parameters
    ----------
    combined_file
        Path to the combined radiance file to read from

    plot_type
        Which type of radiance plot to make; can be a string convertible to a :class:`RadiancePlotType` or an instance of the latter.

    boxes
        A list of 4-element sequences giving ``lonmin, lonmax, latmin, latmax`` to plot radiances/residuals from in the combined file.

    radiance_types
        What quantity to use for the radiances or residuals for each column of the grid of radiance/residual plots. Must be a list of 
        strings convertible to a :class:`FmRadianceType` or instances of the latter. Each entry in the list corresponds to one column 
        of the grid of plots. The default is to use all quantities codified by :class:`FmRadianceType`.

    image
        The image to use as the background for the map. The default is to use Cartopy's stock image, but you can pass any subclass
        of :class:`images.ImageProvider`.

    day_flag
        When ``1`` (default) only daytime values are plotted. Can set to ``0`` to only plot nighttime or ``None``
        to plot both (if present in the file). Requires ``dnf_file`` if this is not ``None``.

    dnf_file
        Path to an L2 file that has day/night flags for the soundings stored in the radiance ``combined_file``. Note that no checking of
        sounding IDs is done currently.

    filterer
        A quality filterer that will both be used to filter the Xgas values plotted and to plot the individual variables
        or metrics used in the filtering. Note that this must be a :class:`quality.MultipleQualityFilterer`, if only a
        single filter is needed, it must be wrapped in one of these.

    filter_near_zero_obs
        Set to ``True`` to change any observed radiances less than 1 K brightness temperature to NaNs. 

    map_extent
        The lonmin, lonmax, latmin, and latmax to use for the map or the string "global" to use a full global map.

    rad_plot_kws
        Additional keywords are passed as the ``plot_kws`` in each call to :func:`plot_radiance_residuals`.
    """
    nbox = len(boxes)
    nx = len(radiance_types)
    ny = nbox + 2
    fig = plt.figure(figsize=(6*nx, 4*ny))
    gs = GridSpec(ncols=nx, nrows=ny, hspace=0.35, wspace=0.35)

    return_axs = dict()

    return_axs['map'] = map_ax = fig.add_subplot(gs[:2, :], projection=ccrs.PlateCarree())
    image.add_image(map_ax)
    if map_extent == 'global':
        map_ax.set_global()
    else:
        map_ax.set_extent(map_extent)


    pbar = miscutils.ProgressBar(nbox*nx, prefix='Plotting')
    for ibox, box in enumerate(boxes, start=1):
        x1, x2, y1, y2 = box
        map_ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], label=f'Box {ibox}', transform=ccrs.PlateCarree())
        return_axs[f'box{ibox}'] = axs = [fig.add_subplot(gs[ibox+1, i]) for i in range(nx)]
        for ifm, fmtype in enumerate(radiance_types):
            pbar.print_bar()
            ax = axs[ifm]
            ax.set_title(f'Box {ibox}')
            plot_radiance_residuals(
                combined_file=combined_file,
                plot_type=plot_type,
                fm_radiances=fmtype,
                day_flag=day_flag,
                dnf_file=dnf_file, 
                box=box, 
                filterer=filterer,
                filter_near_zero_obs=filter_near_zero_obs,
                ax=ax,
                **rad_plot_kws
            )
        map_ax.legend()
    return return_axs


def _add_features(map_features, ax):
    for feature in map_features:
        if isinstance(feature, (str, Path)):
            jgeo.plot_shapes_from_csv(feature, ax=ax)
        else:
            ax.add_feature(feature)
