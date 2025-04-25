"""Reader functions to organize data specifically for ML training and validation.
"""

from jllutils import miscutils
import jllutils.geography as jgeo
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
import xarray as xr
from warnings import warn

from muses_utils.atmosphere import column_integrate, calculate_xvmr

from typing import Dict, Optional, Sequence, Union


def load_xpan_dataframe(keys: Sequence[str], land_only: bool = True, airs_l2_files: Optional[dict] = None,
                        cris_l2_files: Optional[dict] = None, apply_cris_h2o_filter: bool = True,
                        do_ak_correction: bool = True) -> pd.DataFrame:
    """Load a dataframe containing the AIRS and CrIS data.

    This dataframe will contain the XPAN800 values, geolocation information, and
    quality values on the CrIS interpolation (e.g. min/max distance and time difference).

    Parameters
    ----------
    keys
        The list of test keys to load for this dataframe.

    land_only
        If ``True``, only soundings over land are returned.

    airs_l2_files
        A dictionary with keys matching ``keys`` and paths to the AIRS combined L2 
        product files as values. If not given, the default files are obtained from
        :func:`test_cases_by_key`.

    cris_l2_files
        A dictionary with keys matching ``keys`` and paths to the CrIS combined L2 
        product files as values. If not given, the default files are obtained from
        :func:`test_cases_by_key`.

    do_ak_correction
        When ``True``, use the AIRS and CrIS averaging kernels to calculate what
        AIRS would retrieve given the CrIS profile.

    Returns
    -------
    data
        A dataframe, indexed by sounding ID, with the AIRS and CrIS data.

    Notes
    -----
    This function will compute XPAN800 if pointed to AIRS files that do not contain that precalculated,
    so it might take a few minutes to load large datasets.
    """
    if airs_l2_files is None:
        airs_l2_files = test_cases_by_key('airs', file='default')
    if cris_l2_files is None:
        cris_l2_files = test_cases_by_key('cris', file='default')
    dfs = []
    for key in keys:
        print(f'Loading {key}')
        # To support including the H2O-1 quality flag, it's easiest to keep
        # all the soundings and then cut down the dataframe after the H2O-1
        # quality is added. I didn't just make that part of _load_muses_dataset
        # because that's really meant to load a single file. 
        airs_df = _load_muses_dataset(airs_l2_files[key], land_only=False)
        airs_h2o_file = find_h2o_file_for_bias_corr(airs_l2_files[key])
        with ncdf.Dataset(airs_h2o_file) as ds:
            airs_df['h2o_qualflag'] = _get_nc_var(ds, 'Quality')
        if land_only:
            xx_keep = airs_df['landflag'] == 1
            airs_df = airs_df.loc[xx_keep, :]

        cris_df = load_original_cris_xpan(cris_l2_files[key], land_only=land_only)
        npre = cris_df.shape[0]
        xx_cris = cris_df.qualflag == 1
        if apply_cris_h2o_filter:
            xx_cris &= (cris_df.h2o_qualflag == 1)
            print('Including H2O-1 quality flag in CrIS filtering')
        else:
            print('NOT Including H2O-1 quality flag in CrIS filtering')
        cris_df = cris_df.loc[xx_cris, :]
        npost = cris_df.shape[0]
        print(f'CrIS has {npre} soundings before quality filtering, {npost} after')

        # Next; match AIRS soundings to their nearest CrIS sounding, then merge.
        match_df = match_airs_to_cris(airs_df, cris_df)
        airs_df = airs_df.rename(columns=lambda c: f'{c}_airs')
        airs_df['test_key'] = key
        merged_df = pd.merge(match_df, airs_df, left_index=True, right_index=True)
        merged_df = pd.merge(merged_df, cris_df.rename(columns=lambda c: f'{c}_cris'), left_on='cris_sid', right_index=True)

        # If requested, compute what AIRS would retrieve given the CrIS posterior
        # profile.
        if do_ak_correction:
            cris_h2o_corr_ratio = cris_df.xpan / cris_df.xpan_no_bc
            airs_ak_dset = _load_ak_dataset(airs_l2_files[key])
            cris_ak_dset = _load_ak_dataset(cris_l2_files[key])
            rodgers_df = _compute_rodgers_xcols(
                match_df=match_df, airs_ak_dset=airs_ak_dset, cris_ak_dset=cris_ak_dset,
                cris_bc_ratio=cris_h2o_corr_ratio
            )
            merged_df = pd.merge(merged_df, rodgers_df, left_index=True, right_index=True)
        dfs.append(merged_df)
        print('')

    return pd.concat(dfs, axis=0)


def load_original_cris_xpan(cris_file: str, land_only: bool = True) -> pd.DataFrame:
    """Load XPAN800 from a non-interpolated CrIS combined file.

    Parameters
    ----------
    cris_file
        Path to the CrIS ``Products_L2-PAN`` or ``Lite_Products_L2-PAN`` file to load.

    land_only
        Whether to only include land soundings.

    Returns
    -------
    dataframe
        A dataframe containing the XPAN800 values, various flags, geolocation data, and the
        H2O column amount used to bias correct the XPAN.
    """
    # The H2O bias correction is simpler if we keep all of the soundings to begin with,
    # so we'll manually handle filtering for land at the end of this function.
    df = _load_muses_dataset(cris_file, land_only=False)
    h2o_file = find_h2o_file_for_bias_corr(cris_file)
    with ncdf.Dataset(h2o_file) as ds:
        c, h2o = compute_h2o_bias_corr(ds, pan_sids=df.index.to_numpy())
        h2o_qual = _get_nc_var(ds, 'Quality')
    df.rename(columns={'xpan': 'xpan_no_bc'}, inplace=True)
    df['xpan'] = df['xpan_no_bc'] + c
    df['h2o_column'] = h2o
    df['h2o_correction'] = c
    df['h2o_qualflag'] = h2o_qual
    if land_only:
        xx_keep = df['landflag'] == 1
        df = df.loc[xx_keep, :]
    return df


def match_airs_to_cris(airs_df: pd.DataFrame, cris_df: pd.DataFrame) -> pd.DataFrame:
    """Find the CrIS soundings closest in distance to each AIRS sounding

    Parameters
    ----------
    airs_df
        Dataframe with at least AIRS sounding longitude and latitude as the "lon" and "lat" columns,
        and AIRS sounding IDs as the index.

    cris_df
        Dataframe with at least CrIS sounding longitude and latitude as the "lon" and "lat" columns,
        and CrIS sounding IDs as the index.

    Returns
    -------
    match_df
        A dataframe with the AIRS sounding IDs as the index, the sounding ID of the closest CrIS
        sounding, the distance between the AIRS and CrIS soundings (in kilometers), and the difference
        in time between the AIRS and CrIS soundings (in seconds) as columns.
    """
    match_df = pd.DataFrame({'cris_sid': '', 'distance': np.nan, 'time_diff': np.nan}, index=airs_df.index)
    if 'qualflag' in cris_df.columns and np.any(cris_df.qualflag.to_numpy() != 1):
        warn('Warning: CrIS dataframe source for interpolation contains poor quality retrievals!')
    cris_lon = cris_df['lon'].to_numpy()
    cris_lat = cris_df['lat'].to_numpy()

    pbar = miscutils.ProgressBar(airs_df.shape[0], prefix='Matching AIRS to nearest CrIS')
    for airs_sid, airs_row in airs_df.iterrows():
        pbar.print_bar()
        airs_lon = airs_row['lon']
        airs_lat = airs_row['lat']
        dists = jgeo.great_circle_distance(airs_lon, airs_lat, cris_lon, cris_lat)
        idx = np.argmin(dists)
        match_df.loc[airs_sid, 'cris_sid'] = cris_df.index[idx]
        match_df.loc[airs_sid, 'distance'] = dists[idx]
        match_df.loc[airs_sid, 'time_diff'] = np.abs(cris_df.timestamp.iloc[idx] - airs_row['timestamp'])

    return match_df


def _load_ak_dataset(l2_product_file: os.PathLike) -> xr.Dataset:
    """Load the variables necessary to do an AK correction as an :class:`xarray.Dataset`.

    ``l2_product_file`` must be a path to a MUSES combined L2 file.
    """
    with ncdf.Dataset(l2_product_file) as ds:
        data = dict()
        sids = _convert_sounding_ids(_get_nc_var(ds, 'SoundingID'))
        data['aks'] = _get_nc_var(ds, 'AveragingKernel')
        data['prior'] = _get_nc_var(ds, 'ConstraintVector')
        data['posterior'] = _get_nc_var(ds, 'Species')
        data['pressure'] = _get_nc_var(ds, 'Pressure')
        data['density'] = _get_nc_var(ds, 'Retrieval/AirDensity')
        data['altitude'] = _get_nc_var(ds, 'Altitude')

    nlev = data['pressure'].shape[1]
    levels = np.arange(nlev, dtype=int)
    for k, v in data.items():
        if v.ndim == 2:
            dims = ('sounding', 'ret_level')
            coords = (sids, levels)
        elif v.ndim == 3:
            dims = ('sounding', 'ret_level', 'true_level')
            coords = (sids, levels, levels)
        else:
            raise NotImplementedError(f'v.ndim = {v.ndim}')

        data[k] = xr.DataArray(v, dims=dims, coords=coords)

    return xr.Dataset(data)


def _compute_rodgers_xcols(match_df: pd.DataFrame, airs_ak_dset: xr.Dataset, cris_ak_dset: xr.Dataset,
                           cris_bc_ratio: Optional[pd.Series] = None, skip_different_priors: bool = False) -> pd.DataFrame:
    """Apply the Rodgers AK equations to calculate what AIRS would retrieve if it observed the CrIS posterior profile.

    Parameters
    ----------
    match_df
        Dataframe from :func:`match_airs_to_cris` that maps AIRS sounding IDs to the nearest CrIS sounding ID.

    airs_ak_dset
        Xarray dataset from :func:`load_ak_dataset` containing the AIRS data.

    cris_ak_dset
        Xarray dataset from :func:`load_ak_dataset` containing the CrIS data.

    cris_bc_ratio
        If given, a series indexed by the CrIS sounding IDs containing ratios to multiply
        the CrIS posterior profiles by before applying the AK correction.

    skip_different_priors
        If ``True``, then soundings priors with too large a difference after the first
        non-fill level between AIRS and CrIS will have NaNs for the computed retrieved
        column average and a flag of 1.

    Returns
    -------
    ak_df
        A dataframe with columns:

        - "c_hat_pwak": the expected retrieved column calulated using the py-tropess-like
          pressure-weighted column AKs.
        - "c_hat_pwak_bc": same as "c_hat_pwak", but with the CrIS posterior scaled by the
          ratio in ``cris_bc_ratio`` for that sounding before applying the AKs.
        - "c_hat_akmat": the expected retrieved column calculated using the full profile
          AK matrix, then integrating the result.
        - "ak_error_flag": will be 0 for successful soundings, -1 for soundings where either AIRS
          or CrIS had all NaNs in their posterior profile, and 1 for soundings where the
          AIRS and CrIS priors were too different (if ``skip_different_priors`` is ``True``).
    """
    pbar = miscutils.ProgressBar(match_df.shape[0], prefix='Calculating c_hat values')
    out_df = pd.DataFrame({
        'c_hat_pwak': np.nan,
        'c_hat_pwak_bc': np.nan,
        'c_hat_akmat': np.nan,
        'ak_error_flag': 0
    }, index=match_df.index)
    for airs_sid, match_row in match_df.iterrows():
        pbar.print_bar()
        p_airs = airs_ak_dset.sel(sounding=airs_sid).pressure.data
        x_airs = airs_ak_dset.sel(sounding=airs_sid).posterior.data
        d_airs = airs_ak_dset.sel(sounding=airs_sid).density.data
        z_airs = airs_ak_dset.sel(sounding=airs_sid).altitude.data
        xa_airs = airs_ak_dset.sel(sounding=airs_sid).prior.data
        ak_matrix_airs = airs_ak_dset.sel(sounding=airs_sid).aks.data

        cris_sid = match_row['cris_sid']
        p_cris = cris_ak_dset.sel(sounding=cris_sid).pressure.data
        x_cris = cris_ak_dset.sel(sounding=cris_sid).posterior.data
        xa_cris = cris_ak_dset.sel(sounding=cris_sid).prior.data

        if np.all(np.isnan(x_airs)) or np.all(np.isnan(x_cris)):
            out_df.loc[airs_sid, 'ak_error_flag'] = -1
            print(f'\nAt least one of AIRS {airs_sid} and CrIS {cris_sid} have all fills, skipping')
            continue

        if not _check_priors_consistent(xa_airs, xa_cris):
            out_df.loc[airs_sid, 'ak_error_flag'] = 1
            if skip_different_priors:
                print(f'\nAIRS {airs_sid} and CrIS {cris_sid} have different priors, skipping')
                continue

        # First just get the x_col_ft AKs for AIRS. Have to deal with
        # fills here because our AK function expects -999s instead of NaNs.
        ok_airs = ~np.isnan(x_airs)
        pw_ak_airs = _compute_x_col_ft_ak(ak_matrix_airs[ok_airs,:][:,ok_airs], p_airs[ok_airs])

        # Apply AIRS AKs to CrIS posterior (interpolating CrIS posterior to the AIRS pressures)
        # We'll try this two different ways: we'll use the pressure-weighted column AK, which
        # does a slightly different integration than py-retrieve, and we'll use the AK matrix
        # to get an x hat, and integrate that.
        ok_cris = ~np.isnan(x_cris)
        interpolator = interp1d(p_cris[ok_cris], x_cris[ok_cris], fill_value='extrapolate')
        x_cris_interp = interpolator(p_airs[ok_airs])
        dc = 1e9 * pw_ak_airs @ (x_cris_interp - xa_airs[ok_airs])
        ca = _do_column_integrate(xa_airs[ok_airs], d_airs[ok_airs], z_airs[ok_airs], p_airs[ok_airs])
        out_df.loc[airs_sid, 'c_hat_pwak'] = ca + dc

        dx = ak_matrix_airs[ok_airs,:][:,ok_airs] @ (x_cris_interp - xa_airs[ok_airs])
        out_df.loc[airs_sid, 'c_hat_akmat'] = _do_column_integrate(
            xa_airs[ok_airs] + dx,
            d_airs[ok_airs], z_airs[ok_airs], p_airs[ok_airs]
        )

        # If a bias correction ratio is supplied, scale the CrIS posterior by that and calculate
        # the corresponding column average.
        if cris_bc_ratio is not None:
            this_r = cris_bc_ratio.loc[cris_sid]
            x_cris_scaled = x_cris_interp * this_r
            dc = 1e9 * pw_ak_airs @ (x_cris_scaled - xa_airs[ok_airs])
            ca = _do_column_integrate(xa_airs[ok_airs], d_airs[ok_airs], z_airs[ok_airs], p_airs[ok_airs])
            out_df.loc[airs_sid, 'c_hat_pwak_bc'] = ca + dc


    return out_df


def _compute_x_col_ft_ak(profile_ak_matrix, pressure_vec, return_with_fills=True, return_debug_info=False):
    pres_range = (215.0, 825.1)
    missing_value = -999.0

    # I think we want to match how we subset in the new XPAN800 calculation, rather than exactly following my
    # current version of py-tropess. I'm also going to do the calculation how I think it should be (i.e. 
    # not subsetting the AK matrix columns).
    ind_not_fill = np.flatnonzero(~np.isclose(pressure_vec, missing_value))
    pres_good = pressure_vec[ind_not_fill]
    i_bottom = np.argmin(np.abs(pres_good - pres_range[1]))
    i_top = np.argmin(np.abs(pres_good - pres_range[0]))
    indp = slice(ind_not_fill[i_bottom], ind_not_fill[i_top+1])

    _, pwf_level, _ = calculate_xvmr(np.ones_like(pressure_vec[indp]), pressure_vec[indp])

    tmp = pwf_level @ profile_ak_matrix[indp,:][:, ind_not_fill]
    if return_with_fills:
        xcol_ak = np.full(pressure_vec.shape, missing_value)
        xcol_ak[ind_not_fill] = tmp
    else:
        xcol_ak = tmp

    if return_debug_info:
        return xcol_ak, {'pwf': pwf_level, 'ak_subset': profile_ak_matrix[indp, :][:, ind_not_fill], 'indp': indp}
    else:
        return xcol_ak


def _do_column_integrate(x, d, z, p):
    not_fill = ~np.isnan(x)
    i_min = np.argmin(np.abs(p[not_fill] - 825))
    i_max = np.argmin(np.abs(p[not_fill] - 215))
    result = column_integrate(
        VMRIn=x[not_fill],
        airDensityIn=d[not_fill],
        altitudeIn=z[not_fill],
        linearFlag=True,
        pressure=p[not_fill],
        minIndex=i_min,
        maxIndex=i_max
    )
    return result['column'] / result['columnAir'] * 1e9


def _check_priors_consistent(xa1, xa2):
    not_fill = (~np.isnan(xa1)) & (~np.isnan(xa2))
    # The first non-fill level may differ because the pressure
    # moves a bit (probably following the terrain). So that means
    # the prior at that level might actually need to be different.
    dxa = np.abs(xa1 - xa2) / xa1
    dxa = dxa[not_fill][1:]
    # I'll allow 1% differences in the prior, because it looks like
    # two priors that are essentially identical when plotted 
    # differ by 0.001% amounts.
    return np.all(dxa < 0.01)


def load_and_merge_quality_vars(orig_df: pd.DataFrame, airs_combine_dirs: Optional[dict] = None) -> pd.DataFrame:
    """Load the variables MUSES uses for quality filtering and merge with an existing dataframe.

    Parameters
    ----------
    orig_df
        The dataframe containing existing data to merge with, normally the one loaded by :func:`load_xpan_dataframe`.
        Must be indexed by the MUSES sounding ID strings.

    airs_combine_dirs
        A dictionary index by date strings in YYYY-MM-DD format with paths to the AIRS directories with the combined
        L2 and radiance product files. If not given, this is obtained from :func:`test_cases_by_key`.

    Returns
    -------
    dataframe
        A dataframe with the same index as ``orig_df``, with the various variables used by MUSES for quality
        filtering added as new columns. They may be NAs if (1) they were fill values in the combined files or
        (2) the combined file was missing a sounding ID present in the ``orig_df`` index.
    """
    keys = _get_test_keys_from_df(orig_df)
    if airs_combine_dirs is None:
        airs_combine_dirs = test_cases_by_key('airs', None)

    qual_df = []
    for key in keys:
        combine_dir = Path(airs_combine_dirs[key])
        l2_file = combine_dir / 'Products_L2-PAN-0.nc'
        rad_file = combine_dir / 'Products_Radiance-PAN.nc'

        rad_vars = {
            'radianceResidualRMS': 'RADIANCERESIDUALRMS',
            'radianceResidualMean': 'RADIANCERESIDUALMEAN',
            'CLOUD_MEAN': 'CLOUDOPTICALDEPTH', # weird that cloud optical depth is in the radiance file...
        }
        rad_df = _load_helper(rad_file, rad_vars, 'SOUNDINGID')

        # TSUR_vs_Apriori and TSUR-Tatm[0] both seem to be unused, as write_quality_flags.py
        # assigns them values of 0 in the values list. Can't find Calscale Mean, Emission Layer,
        # O3_Ccurve, O3_Slope_QA, O3_tropo_consistency in the files. Skipped Deviation_QA because
        # it's not clear which one it is.
        l2_vars = {
            'ResidualNormInitial': 'Characterization/ResidualNormInitial',
            'ResidualNormFinal': 'Characterization/ResidualNormFinal',
            'RadianceMaximumSNR': 'Characterization/RadianceMaximumSNR',
            'KdotDL': 'Characterization/KDotDL_QA', # dot product of Jacobian and radiance residual
            'LdotDL': 'Characterization/LDotDL_QA', # dot product of radiance and radiance residual
            'PCLOUD': 'Retrieval/CloudTopPressure',
            'CLOUD_VAR': 'Characterization/CloudVariability_QA',
            'EMIS_MEAN': 'Characterization/SurfaceEmissMean_QA',
            'Desert_Emiss_QA': 'Characterization/Desert_Emiss_QA',
            'H2O_H2O_Quality': 'Characterization/H2O_H2O_Corr_QA',
            'TATM_Propagated': 'Characterization/Propagated_TATM_QA',
            'O3_Propagated': 'Characterization/Propagated_O3_QA',
            'H2O_Propagated': 'Characterization/Propagated_H2O_QA',
        }
        l2_df = _load_helper(l2_file, l2_vars, 'SoundingID')

        this_qual_df = pd.merge(l2_df, rad_df, left_index=True, right_index=True)
        assert this_qual_df.shape[0] == l2_df.shape[0], 'Not all L2 data got associated radiance data.'
        qual_df.append(this_qual_df)

    qual_df = pd.concat(qual_df, axis=0)
    return pd.merge(orig_df, qual_df, left_index=True, right_index=True, how='left')


def load_and_merge_radiance_vars(orig_df: pd.DataFrame, airs_combine_dirs: Optional[dict] = None) -> pd.DataFrame:
    """Load the observed, fit, and residual radiances as a dataframe and merge them with an existing dataframe.

    Parameters
    ----------
    orig_df
        The dataframe containing existing data to merge with, normally the one loaded by :func:`load_xpan_dataframe`.
        Must be indexed by the MUSES sounding ID strings.

    airs_combine_dirs
        A dictionary index by date strings in YYYY-MM-DD format with paths to the AIRS directories with the combined
        L2 and radiance product files. If not given, this is obtained from :func:`test_cases_by_key`.

    Returns
    -------
    dataframe
        A dataframe with the same index as ``orig_df``, with the various radiance quantities added as new columns.
        Each frequency will have its own column. The radiance values may be NAs if (1) they were fill values in
        the combined files or (2) the combined file was missing a sounding ID present in the ``orig_df`` index.
    """
    keys = _get_test_keys_from_df(orig_df)
    if airs_combine_dirs is None:
        airs_combine_dirs = test_cases_by_key('airs', None)

    rad_df = []
    for key in keys:
        combine_dir = Path(airs_combine_dirs[key])
        rad_file = combine_dir / 'Products_Radiance-PAN.nc'

        with ncdf.Dataset(rad_file) as ds:
            sid = _convert_sounding_ids(_get_nc_var(ds, 'SOUNDINGID'))
            rad_fit = _get_nc_var(ds, 'RADIANCEFIT')
            rad_obs = _get_nc_var(ds, 'RADIANCEOBSERVED')
            rad_freq = _get_nc_var(ds, 'FREQUENCY')

        # Must be on a consistent frequency grid to work
        freq_diff = np.diff(rad_freq, axis=0)
        freq_diff = freq_diff[np.isfinite(freq_diff)]
        assert np.allclose(freq_diff, 0), 'Frequency grid must be consistent'

        bt_fit = _to_brightness_temperature(rad_freq, rad_fit)
        bt_obs = _to_brightness_temperature(rad_freq, rad_obs)
        bt_resid = bt_fit - bt_obs
        # Taking the first row without NaNs is more relable than a mean at getting a consistent 
        # frequency grid.
        i = np.flatnonzero(np.all(~np.isnan(rad_freq), axis=1))[0]
        freq_grid = rad_freq[i]

        df_dict = dict()
        for i, freq in enumerate(freq_grid):
            df_dict[f'bt_fit_{freq:.1f}'] = bt_fit[:,i]
            df_dict[f'bt_obs_{freq:.1f}'] = bt_obs[:,i]
            df_dict[f'bt_resid_{freq:.1f}'] = bt_resid[:,i]
        rad_df.append(pd.DataFrame(df_dict, index=sid))

    rad_df = pd.concat(rad_df, axis=0)
    return pd.merge(orig_df, rad_df, left_index=True, right_index=True, how='left')


def _load_muses_dataset(combined_file, land_only=True):
    with ncdf.Dataset(combined_file) as ds:
        xpan_found = 'XPAN800' in ds.groups['Retrieval'].variables.keys()
        vars = {
            'xpan': 'Retrieval/XPAN800' if xpan_found else calc_xpan800,
            'lon': 'Longitude',
            'lat': 'Latitude',
            'timestamp': 'Time',
            'qualflag': 'Quality',
            'dayflag': 'DayNightFlag',
            'landflag': 'LandFlag',
        }
        return _load_helper(combined_file, vars, land_only=land_only)        


def _load_interp_cris(interp_file):
    with ncdf.Dataset(interp_file) as ds:
        data = dict()
        for varname, var in ds.variables.items():
            data[varname] = var[:]
            if varname not in {'sid'}:
                data[varname] = data[varname].filled(np.nan)
            elif varname == 'timestamp':
                data[varname] = data[varname].filled(-999)

        sid = data.pop('sid')
        return pd.DataFrame(data, index=sid)


def calc_xpan800(combined_ds, xx_keep=None):
    vmr = _get_nc_var(combined_ds, 'Species')
    air_dens = _get_nc_var(combined_ds, 'Retrieval/AirDensity')
    alt = _get_nc_var(combined_ds, 'Altitude')
    pres = _get_nc_var(combined_ds, 'Pressure')
    xpan = np.full(vmr.shape[0], -999.0)
    if xx_keep is None:
        xx_keep = np.ones(vmr.shape[0], dtype=bool)

    pbar = miscutils.ProgressBar(vmr.shape[0], prefix='Calculating XPAN800')
    for i in range(vmr.shape[0]):
        pbar.print_bar()
        zz = vmr[i] > -990.
        if np.any(zz) and xx_keep[i]:
            minIndex = np.argmin(np.abs(pres[i][zz] - 825))
            maxIndex = np.argmin(np.abs(pres[i][zz] - 215))
            result = column_integrate(
                VMRIn=vmr[i][zz],
                airDensityIn=air_dens[i][zz],
                altitudeIn=alt[i][zz],
                linearFlag=True,
                pressure=pres[i][zz],
                minIndex=minIndex,
                maxIndex=maxIndex,
                #minPressure=np.asarray([200]),
                #maxPressure=np.asarray([800])
            )
            xpan[i] = result['column'] / result['columnAir'] * 1e9
    return xpan[xx_keep]


def _load_helper(file, variables, sid_var='SoundingID', land_only=True):
    if not isinstance(variables, dict):
        variables = {v: v for v in variables}

    with ncdf.Dataset(file) as ds:
        sounding_ids = _convert_sounding_ids(_get_nc_var(ds, sid_var))
        if land_only and 'LandFlag' in ds.variables:
            land_flag = _get_nc_var(ds, 'LandFlag')
            xx_keep = land_flag == 1
            sounding_ids = sounding_ids[xx_keep]
        else:
            xx_keep = None

        data = dict()
        for key, var in variables.items():
            if isinstance(var, str): 
                data[key] = _get_nc_var(ds, var)
                if xx_keep is not None:
                    data[key] = data[key][xx_keep]
            else:
                data[key] = var(ds, xx_keep)
            assert data[key].ndim == 1, f'{key} is not 1D, is actually {data[key].ndim}D'

    return pd.DataFrame(data, index=sounding_ids)

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


def compute_h2o_bias_corr(h2o_ds, pan_sids=None):
    if pan_sids is not None:
        h2o_sids = _convert_sounding_ids(_get_nc_var(h2o_ds, 'SoundingID'))
        assert np.array_equal(h2o_sids, pan_sids), 'H2O file and PAN file have different sounding IDs'
    h2o_column = _get_nc_var(h2o_ds, 'Retrieval/Column')[:,0]
    return 0.05 + 0.035e-23 * h2o_column, h2o_column


def iter_test_cases(root_dir, file=None, must_exist=False):
    if root_dir == 'airs':
        root_dir = Path('data/validation/airs-runs/airs/')
        file = 'Products_L2-PAN-0.nc' if file == 'default' else file
    elif root_dir == 'cris':
        root_dir = Path('data/validation/airs-runs/cris/')
        file = 'Products_L2-PAN-0.nc' if file == 'default' else file
    else:
        root_dir = Path(root_dir)

    cases = [
        ('aus-jan01', '2020-01-01', 'Australian_Fires_PAN'),
        ('aus-jan05', '2020-01-05', 'Australian_Fires_PAN'),
        ('wcf-sep11', '2020-09-11', 'West_Coast_Fires_Pacific'),
        ('ama-sep11', '2020-09-11', 'Amazon_Fires_PAN'),
        ('afr-sep11', '2020-09-11', 'African_Fires_PAN'),
        ('wcf-sep13', '2020-09-13', 'West_Coast_Fires_Pacific'),
        ('glo-jun10', '2023-06-10', 'Global_Survey_Grid_0.7'),
        ('glo-jun13', '2023-06-13', 'Global_Survey_Grid_0.7')
    ]

    for key, date, profile in cases:
        test_path = root_dir / date / 'combine' / profile
        if file is not None:
            test_path = test_path / file

        if not test_path.exists() and must_exist:
            err_file = 'combine directory' if file is None else file
            raise IOError(f'{err_file} not found for {date} {profile} (path = {test_path})')
        yield key, test_path


def test_cases_by_key(root_dir, file=None) -> Dict[str, Path]:
    return {k: p for k, p in iter_test_cases(root_dir, file=file)}


def _get_dates_from_sid_index(df):
    dates = pd.to_datetime(df.index.str[:8].unique())
    return [d.strftime('%Y-%m-%d') for d in dates]

def _get_test_keys_from_df(df) -> Sequence[str]:
    return list(df.test_key.unique())

def _get_nc_var(ds: ncdf.Dataset, variable: str, fill=-999):
    data = ds[variable][:]
    is_int = np.issubdtype(data.dtype, np.integer)
    data = data.filled(fill if is_int else np.nan)
    if fill is not None and not is_int:
        data[np.isclose(data, fill)] = np.nan
    return data


def _convert_sounding_ids(sounding_ids: np.ndarray) -> Union[str, np.ndarray]:
    sounding_ids = sounding_ids.astype('uint8')
    if sounding_ids.ndim == 1:
        return sounding_ids.tobytes().decode()
    else:
        return np.array([r.tobytes().decode() for r in sounding_ids])

def _to_brightness_temperature(frequency: np.ndarray, rad: np.ndarray) -> np.ndarray:
    PLANCK = 6.626176E-27
    CLIGHT = 2.99792458E+10
    BOLTZ = 1.380662E-16
    RADCN1 = 2. * PLANCK * CLIGHT * CLIGHT*1.E-7
    RADCN2 = PLANCK * CLIGHT / BOLTZ

    bt = RADCN2 * frequency / np.log(1 + (RADCN1 * frequency**3 / rad))
    return bt
