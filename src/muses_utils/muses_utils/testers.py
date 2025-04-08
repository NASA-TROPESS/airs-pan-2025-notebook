from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import warnings

from jllutils import miscutils

from . import readers as mread


class Abstract2DReduction(ABC):
    @abstractmethod
    def reduce(self, delta):
        pass

    @property
    @abstractmethod
    def prefix(self):
        pass


class MaxAbsReduce2D:
    def reduce(self, delta):
        shape = np.shape(delta)
        delta = np.reshape(delta, (shape[0], shape[1], -1))
        return np.nanmax(np.abs(delta), axis=2)
    
    @property
    def prefix(self):
        return 'Max. abs.'


def check_variables(ds1, ds2, _record=None, _prefix='/'):
    """Check that variables and their data are the same between two netCDF datasets
    
    Parameters
    ----------
    ds1, ds2 : netCDF4.Dataset
        The two handles to netCDF datasets to compare.

    _record, _prefix
        Inputs used for internal recursive calls.

    Returns
    -------
    ok : bool
        True if all variables present in ``ds1`` are in ``ds2`` and vice versa and all are equal.
        For floating point types, the arrays' unmasked values must be within default floating point
        uncertainty in :func:`numpy.ma.allclose`. For other types, the arrays must be equal by
        :func:`numpy.ma.allequal`. In all cases, the masking arrays must be identical.

    record : dict
        A record of how each variable compared. Full internal paths to variables are used for keys.
        The values will be one of "matches" (same in both files), "differs" (in both files but not equal),
        "missing" (in ``ds1`` but not ``ds2``), or "new" (in ``ds2`` but not ``ds1``).
    """
    if _record is None:
        _record = dict()
        
    ok = True
    
    # First check that all variables in the original dataset are present in the new
    # one and have the same values
    for varname, varobj in ds1.variables.items():
        fullname = f'{_prefix}{varname}'
        if varname not in ds2.variables:
            ok = False
            _record[fullname] = 'missing'
        elif np.issubdtype(varobj.dtype, np.floating):
            if np.ma.allclose(varobj[:], ds2[varname][:]) and np.array_equal(varobj[:].mask, ds2[varname][:].mask):
                _record[fullname] = 'matches'
            else:
                _record[fullname] = 'differs'
                ok = False
        else:
            if np.ma.allequal(varobj[:], ds2[varname][:]) and np.array_equal(varobj[:].mask, ds2[varname][:].mask):
                _record[fullname] = 'matches'
            else:
                _record[fullname] = 'differs'
                ok = False
                
    # Then check if the new dataset has any new variables
    for varname in ds2.variables.keys():
        fullname = f'{_prefix}{varname}'
        if varname not in ds1.variables:
            _record[fullname] = 'new'
            ok = False
            
    # Now do groups - for every group in common, recurse. For groups not present in
    # both files, record that.
    for grpname, grpobj in ds1.groups.items():
        fullname = f'{_prefix}{grpname}/'
        if grpname not in ds2.groups:
            ok = False
            _record[fullname] = 'missing_group'
        else:
            this_ok, _ = check_variables(grpobj, ds2.groups[grpname], _record=_record, _prefix=fullname)
            ok = ok and this_ok
            
    for grpname in ds2.groups.keys():
        fullname = f'{_prefix}{grpname}/'
        if grpname not in ds1.groups:
            ok = False
            _record[fullname] = 'new_group'
                
    return ok, _record


def report_file_differences(ds1, ds2):
    ok, record = check_variables(ds1, ds2)
    if ok:
        print(f'Files match!!! ({ds1.filepath()} and {ds2.filepath()})')
    else:
        print('Files DIFFER! Offending variables are:')
        for name, status in record.items():
            if status != 'matches':
                print(f'{name}: {status}')


def plot_variable_comparison(ds1, ds2, variable, axs=None, reduce_op=MaxAbsReduce2D()):
    """Plot differences between a variable in two datasets.
    
    Parameters
    """
    if axs is None:
        _, axs = plt.subplots(1,2,figsize=(10,5))
        
    data1 = mread.get_nc_var(ds1, variable)
    units1 = mread.get_nc_var(ds1, variable, read_array=False).Units
    data2 = mread.get_nc_var(ds2, variable)
    units2 = mread.get_nc_var(ds2, variable, read_array=False).Units
    if units1 != units2:
        raise ValueError('Units are not equal!')
    
    delta = data2 - data1
    perdelta = delta / data1 * 100
    
    if np.ndim(delta) == 1:
        axs[0].plot(delta, marker='.')
        axs[0].set_ylabel(f'{variable} difference ({units1})')
        axs[0].set_xlabel('Sounding')
        axs[1].plot(perdelta, marker='.')
        axs[1].set_ylabel(f'{variable} percent difference')
        axs[1].set_xlabel('Sounding')
    elif np.ndim(delta) == 2:
        h = axs[0].pcolormesh(delta, norm=mcolors.CenteredNorm(), cmap='seismic')
        axs[0].set_xlabel('Level')
        axs[0].set_ylabel('Sounding')
        plt.colorbar(h, ax=axs[0], label=f'{variable} difference ({units1})')
        h = axs[1].pcolormesh(perdelta, norm=mcolors.CenteredNorm(), cmap='seismic')
        plt.colorbar(h, ax=axs[1], label=f'{variable} percent difference')
        axs[1].set_xlabel('Level')
        axs[1].set_ylabel('Sounding')
    else:
        delta = reduce_op.reduce(delta)
        perdelta = reduce_op.reduce(perdelta)
        h = axs[0].pcolormesh(delta, norm=mcolors.CenteredNorm(), cmap='seismic')
        axs[0].set_xlabel('Level')
        axs[0].set_ylabel('Sounding')
        plt.colorbar(h, ax=axs[0], label=f'{reduce_op.prefix} {variable} difference ({units1})')
        h = axs[1].pcolormesh(perdelta, norm=mcolors.CenteredNorm(), cmap='seismic')
        plt.colorbar(h, ax=axs[1], label=f'{reduce_op.prefix} {variable} percent difference')
        axs[1].set_xlabel('Level')
        axs[1].set_ylabel('Sounding')
        
        
def plot_mismatch_comparison(ds1, ds2, record):
    mismatched_vars = [k for k, v in record.items() if v == 'differs']
    ny = len(mismatched_vars)
    nx = 2
    fig, axs = plt.subplots(ny, nx, figsize=(5*nx, 5*ny))
    if ny == 1:
        axs = [axs]
        
    pbar = miscutils.ProgressBar(ny, prefix='Plotting variable')
    for ivar, misvar in enumerate(mismatched_vars):
        pbar.print_bar()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            plot_variable_comparison(ds1, ds2, misvar, axs=axs[ivar])
    
    plt.subplots_adjust(wspace=0.33)