"""Validation tools for PAN vs. ATom data

The typical use would be:

1. Call `ATomProfileSet.from_dir(...)` on a directory containing ATom profiles with GEOS-Chem profiles
   appended to the top provided to Vivienne by Emily Fisher.
2. Call `load_airs_sounding_df(...)` with the path to the Lite file with the coincident AIRS (or other
   instrument) soundings. Alternately, specify the filter to apply to the soundings.
3. Call `make_comparison_dfs(...)` with (1) the output from `ATomProfileSet.profiles_for_campaign(i)`,
   (where `i` is 1 to 4), (2) either of the dataframes returned by `load_airs_soundings_df`, and (3)
   the same Lite file given as input to `load_airs_soundings_df`
4. Pass one of these dataframes to `plot_xpan_comparison_curtain` or another (future) plotting function
"""
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d

from jllutils import miscutils
import jllutils.plots as jplt
import jllutils.geography as jgeo
from muses_utils import quality, readers

from typing import Optional


ATOM_PROFILE_DIR = Path('/tb/sandbox14/vpayne/retrievals/PAN/Fischer/ATom/profiles/')
ATOM_DIVIDING_LON = -60

class ATomProfileSet:
    def __init__(self, *files):
        self._profiles = []
        
        for file in files:
            self._profiles.extend(self._load_prof_file(file))
            
        self._date_map = dict()
        self._campaign_map = dict()
        for i, p in enumerate(self._profiles):
            pdate = p.index[0].date()
            dinds = self._date_map.setdefault(pdate, [])
            dinds.append(i)
            
            cam_num = p['atom_campaign_num'].iloc[0]
            cinds = self._campaign_map.setdefault(cam_num, [])
            cinds.append(i)
            
    def __getitem__(self, item):
        return self._profiles[item]
            
    @classmethod
    def from_dir(cls, directory):
        files = list(Path(directory).glob('*.nc'))
        return cls(*files)
            
    @classmethod
    def _load_prof_file(cls, file):
        dfs = []
        var_names = ['year', 'month', 'day', 'alt', 'pres', 'lat', 'lon', 'pan_ecd', 'pan_gtcims', 'co_noaa', 'co_qcl', 'utc_time']
        with ncdf.Dataset(file) as ds:
            for i in range(ds.dimensions['profiles'].size):
                prof_slice = ds['profile_data'][i]
                xx = np.any(np.isfinite(prof_slice), axis=0)
                prof_slice = prof_slice[:, xx]
                this_df = pd.DataFrame({k: prof_slice[i] for i, k in enumerate(var_names)})
                this_df['is_model'] = this_df['year'].isna() | this_df['alt'].isna()
                this_df = _fill_time(this_df)
                
                this_df.index = pd.to_datetime({'year': this_df['year'], 'month': this_df['month'], 'day': this_df['day']}) + pd.TimedeltaIndex(this_df['utc_time'], unit='s')
                this_df.index.name = 'datetime'
                this_df['atom_campaign_num'] = atom_campaign_num(this_df.index[0])
                dfs.append(this_df)
                
        return dfs
    
    def profiles_for_date(self, date):
        date = pd.to_datetime(date).date()
        try:
            inds = self._date_map[date]
        except KeyError:
            raise ValueError(f'No profiles for {date}')
        else:
            return [self._profiles[i] for i in inds]
        
    def profiles_for_campaign(self, atom_campaign_num: int):
        try:
            inds = self._campaign_map[atom_campaign_num]
        except KeyError:
            raise ValueError(f'No profiles for AToM campaign {atom_campaign_num}')
        else:
            return [self._profiles[i] for i in inds]
        
    
    
def atom_campaign_num(date):
    periods = [
        (pd.Timestamp(2016,7,1), pd.Timestamp(2016,8,31)),
        (pd.Timestamp(2017,1,1), pd.Timestamp(2017,2,28)),
        (pd.Timestamp(2017,9,1), pd.Timestamp(2017,10,31)),
        (pd.Timestamp(2018,4,1), pd.Timestamp(2018,5,31))
    ]
    
    for i, (start, stop) in enumerate(periods, start=1):
        if (date >= start) & (date <= stop):
            return i
        
    raise ValueError(f'Date {date} is not in any defined AToM campaign period')
    
    
def _fill_time(this_df):
    for k in ['year', 'month', 'day', 'utc_time']:
        this_df.loc[np.isclose(this_df[k].to_numpy(), 0), k] = np.nan
        this_df[k] = this_df[k].fillna(method='ffill').astype(int)
        
    return this_df
    
    
def load_airs_sounding_df(airs_lite_file: str, filt: Optional[quality.AbstractQualityFilterer] = None) -> pd.DataFrame:
    with ncdf.Dataset(airs_lite_file) as ds:
        sounding_ids = readers.convert_sounding_ids(readers.get_nc_var(ds, 'SoundingID'))
        lon = readers.get_nc_var(ds, 'Longitude')
        lat = readers.get_nc_var(ds, 'Latitude')
        land_ocean = readers.get_nc_var(ds, 'LandFlag')
        time = pd.Timestamp(1993, 1, 1) + pd.TimedeltaIndex(readers.get_nc_var(ds, 'Time'), unit='s')
        
    df = pd.DataFrame({'lon': lon, 'lat': lat, 'is_land': land_ocean == 1, 'time': time}, index=sounding_ids)
    if filt:
        xx_good = filt.quality_flags(airs_lite_file)
        df = df[xx_good]
    return df
        
    
    
def _find_coincident_airs_soundings(profile_df: pd.DataFrame, airs_df: pd.DataFrame, dist_km: float = 50.0, time_diff = pd.Timedelta(hours=9)):
    prof_lon = profile_df['lon'].mean()
    prof_lat = profile_df['lat'].mean()
    prof_time = profile_df.index.mean()
    
    airs_lon = airs_df['lon'].to_numpy()
    airs_lat = airs_df['lat'].to_numpy()
    airs_time = pd.DatetimeIndex(airs_df['time'])
    
    prof_to_airs_dist = jgeo.great_circle_distance(airs_lon, airs_lat, prof_lon, prof_lat)
    xx_dist = prof_to_airs_dist <= dist_km
    prof_to_airs_time = np.abs((airs_time - prof_time).total_seconds())
    xx_time = prof_to_airs_time <= np.abs(time_diff.total_seconds())
    
    return airs_df.index[xx_dist & xx_time]


def _calculate_atom_xpan(profile_df: pd.DataFrame, airs_sounding_df: pd.DataFrame, airs_lite_file: str, min_num_fov: int = 5, req_pres_bounds=(200, 800), debug_exit_cause: bool = False):
    matched_soundings = _find_coincident_airs_soundings(profile_df, airs_sounding_df)
    if matched_soundings.size < min_num_fov:
        if debug_exit_cause:
            print('Not enough soundings')
        return None
    
    with ncdf.Dataset(airs_lite_file) as ds:
        airs_sids = readers.convert_sounding_ids(readers.get_nc_var(ds, 'SoundingID'))
        airs_lat = readers.get_nc_var(ds, 'Latitude')
        airs_lon = readers.get_nc_var(ds, 'Longitude')
        pres_grid = readers.get_nc_var(ds, 'Pressure')
        xpan800 = readers.get_nc_var(ds, 'Retrieval/XPAN800')
        xpan800_ak = readers.get_nc_var(ds, 'Characterization/XPAN800_AK')
        prior_prof = 1e9 * readers.get_nc_var(ds, 'ConstraintVector')  # convert mol/mol -> ppb
        prior_xpan = readers.get_nc_var(ds, 'Retrieval/XPAN800_Prior')
        
    pan_file = Path(airs_lite_file)
    h2o0_file = pan_file.parent / pan_file.name.replace('PAN-0', 'H2O-0')
    h2o1_file = pan_file.parent / pan_file.name.replace('PAN-0', 'H2O-1')
    with ncdf.Dataset(h2o0_file) as ds:
        # The first column of the array is the total column density, others are different sub-columns
        h2o0_cols = readers.get_nc_var(ds, 'Retrieval/Column')[:,0]
    with ncdf.Dataset(h2o1_file) as ds:
        h2o1_cols = readers.get_nc_var(ds, 'Retrieval/Column')[:,0]
        
    atom_ecd_xpan = np.full(matched_soundings.size, np.nan)
    atom_gtcims_xpan = np.full(matched_soundings.size, np.nan)
    
    pmin, pmax = req_pres_bounds
    if pmin > pmax:
        pmin, pmax = pmax, pmin
    
    xx_ecd = ~profile_df['pan_ecd'].isna()
    has_ecd = (profile_df.loc[xx_ecd, 'pres'].max() > 800) and (profile_df.loc[xx_ecd, 'pres'].min() < 200)
    if has_ecd:
        # Last PAN first, because that will match the minimum pressure
        extrap_vals = (profile_df.loc[xx_ecd, 'pan_ecd'].iloc[-1], profile_df.loc[xx_ecd, 'pan_ecd'].iloc[0])
        atom_ecd_interpolator = interp1d(
            np.log(profile_df.loc[xx_ecd, 'pres'].to_numpy()), 
            profile_df.loc[xx_ecd, 'pan_ecd'].to_numpy(),
            fill_value=extrap_vals,
            bounds_error=False
        )
    
    xx_gtcims = ~profile_df['pan_gtcims'].isna()
    has_gtcims = (profile_df.loc[xx_gtcims, 'pres'].max() > 800) and (profile_df.loc[xx_gtcims, 'pres'].min() < 200)
    if has_gtcims:
        # Last PAN first, because that will match the minimum pressure
        extrap_vals = (profile_df.loc[xx_ecd, 'pan_gtcims'].iloc[-1], profile_df.loc[xx_ecd, 'pan_gtcims'].iloc[0])
        atom_gtcims_interpolator = interp1d(
            np.log(profile_df.loc[xx_gtcims, 'pres'].to_numpy()), 
            profile_df.loc[xx_gtcims, 'pan_gtcims'].to_numpy(),
            fill_value=extrap_vals,
            bounds_error=False
        )
        
    if not has_ecd and not has_gtcims:
        if debug_exit_cause:
            print('PAN profiles do not cover enough vertical distance')
        return None
    
    airs_lon_out = []
    airs_lat_out = []
    airs_xpan_out = []
    airs_h2o_column_0_out = []
    airs_h2o_column_1_out = []
    
    for isounding, sounding_id in enumerate(matched_soundings):
        airs_idx = np.flatnonzero(airs_sids == sounding_id).item()
        airs_lon_out.append(airs_lon[airs_idx])
        airs_lat_out.append(airs_lat[airs_idx])
        airs_xpan_out.append(xpan800[airs_idx])
        airs_h2o_column_0_out.append(h2o0_cols[airs_idx])
        airs_h2o_column_1_out.append(h2o1_cols[airs_idx])
        
        this_ak = xpan800_ak[airs_idx]
        this_ln_p = np.log(pres_grid[airs_idx])
        this_xa = prior_prof[airs_idx]
        this_za = prior_xpan[airs_idx]
        
        zz = ~np.isnan(this_ln_p)
        
        # TODO: confirm that the AKs are used correctly
        if has_ecd:
            atom_ecd_prof = atom_ecd_interpolator(this_ln_p[zz])
            atom_ecd_xpan[isounding] = this_za + np.dot(this_ak[zz], atom_ecd_prof - this_xa[zz])
        
        if has_gtcims:
            atom_gtcims_prof = atom_gtcims_interpolator(this_ln_p[zz])
            atom_gtcims_xpan[isounding] = this_za + np.dot(this_ak[zz], atom_gtcims_prof - this_xa[zz])
        
    return pd.DataFrame({
            'ecd_xpan': atom_ecd_xpan,
            'gtcims_xpan': atom_gtcims_xpan,
            'airs_lon': airs_lon_out,
            'airs_lat': airs_lat_out,
            'airs_xpan': airs_xpan_out,
            'h2o_col_0': airs_h2o_column_0_out,
            'h2o_col_1': airs_h2o_column_1_out,
        }, 
        index=matched_soundings)


def make_comparisons_dfs(profile_dfs, airs_sounding_df, airs_lite_file):
    dfs = []
    nskip = 0
    pbar = miscutils.ProgressBar(len(profile_dfs), prefix='Calculating AToM XPAN')
    for iprof, prof_df in enumerate(profile_dfs):
        pbar.print_bar()
        comp_df = _calculate_atom_xpan(prof_df, airs_sounding_df, airs_lite_file)
        if comp_df is not None:
            comp_df['profile_id'] = iprof
            dfs.append(comp_df)
        else:
            nskip += 1
        
    print(f'{nskip}/{len(profile_dfs)} skipped for insufficient soundings or vertical extent')
        
    comparison_df = pd.concat(dfs, axis=0)
    comparison_df['ecd_pan_diff'] = comparison_df['airs_xpan'] - comparison_df['ecd_xpan']
    comparison_df['ecd_pan_perdiff'] = comparison_df['ecd_pan_diff'] / comparison_df['ecd_xpan'].abs() * 100
    comparison_df['gtcims_pan_diff'] = comparison_df['airs_xpan'] - comparison_df['gtcims_xpan']
    comparison_df['gtcims_pan_perdiff'] = comparison_df['gtcims_pan_diff'] / comparison_df['gtcims_xpan'].abs() * 100
        
    mean_comp_df = comparison_df.groupby('profile_id').mean()
    std_comp_df = comparison_df.groupby('profile_id').std()
    for col in ['ecd_xpan', 'gtcims_xpan', 'airs_xpan', 'ecd_pan_diff', 'gtcims_pan_diff', 'ecd_pan_perdiff', 'gtcims_pan_perdiff']:
        mean_comp_df[f'{col}_std'] = std_comp_df[col] 
    
    return mean_comp_df, comparison_df


def plot_xpan_comparison_curtain(comparison_df, use_gtcims=False, use_percent=False, is_mean=None, ax=None, **style):
    if is_mean is None:
        is_mean = comparison_df.index.name == 'profile_id'
        
    pan_str = 'gtcims' if use_gtcims else 'ecd'
    pan_label = 'GT CIMS' if use_gtcims else 'PANTHER'
    diff_str = 'perdiff' if use_percent else 'diff'
    diff_label = r'%$\Delta$' if use_percent else r'$\Delta$'
    diff_unit = '%' if use_percent else 'ppb' 
    pan_col = f'{pan_str}_pan_{diff_str}'
    ylabel = f'{diff_label} PAN (AIRS vs. {pan_label}, {diff_unit})'
    
    style.setdefault('marker', 's')
    style.setdefault('ls', 'none')
    x = comparison_df['airs_lat'].to_numpy()
    y = comparison_df[pan_col].to_numpy()
    
    ax = ax or plt.gca()
    ax.plot(x, y, **style)
    if is_mean:
        e = comparison_df[f'{pan_col}_std']
        jplt.plot_error_bar(ax, x, y, e, color=style.get('color'))
    ax.set_xlabel('Latitude')
    ax.set_ylabel(ylabel)



