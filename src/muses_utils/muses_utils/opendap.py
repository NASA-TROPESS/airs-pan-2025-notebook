from datetime import datetime
from typing import Optional
import numpy as np

from netrc import netrc
import xarray as xr
from pydap.client import open_url
from pydap.cas.urs import setup_session

from jllutils import miscutils
from jllutils.subutils import ncdf as ncio
from .atmosphere import calculate_xvmr

URL_TEMPLATES = {
    'WCF_CRIS_PAN_STD': 'https://tropess.gesdisc.eosdis.nasa.gov/opendap/TROPESS_Special/TRPSDL2PANCRSWCF.1/{date:%Y}/TROPESS_CrIS-SNPP_L2_Standard_PAN_{date:%Y%m%d}_MUSES_R1p9_SC_F0p1.nc'
}

DEFAULT_TROPESS_VARS = {
    'lon': 'longitude',
    'lat': 'latitude',
    'vmr': 'x',
    'pres': 'pressure'
}

def read_tropess_url(url: str, variables: dict = DEFAULT_TROPESS_VARS, date: Optional[datetime] = None, integrate_xgas: str = 'quick', pmax: float = 825, pmin: float = 215):
    """Read TROPESS data from an OpenDAP URL

    Parameters
    ----------
    url
        The OpenDAP URL to download from. Some are available in the ``URL_TEMPLATES`` variable of this module. The easiest way to find others is to go to 
        https://tes.jpl.nasa.gov/tropess/get-data/products, select your product at the bottom of the page, and use the "OPENDAP" button on the right of
        the Earthdata page to navigate to the correct :file:`.nc` file. 

        Date elements in the URL can be replaced with ``{date:FMT}`` where ``FMT`` is a datetime formatting string. If you pass a URL with this in it,
        you must also pass the ``date`` argument.

    variables
        A list or dictionary listing variables to read from the OpenDAP repo. If a list, then it must be variable names in the root
        group of the OpenDAP repo, and the output dictionary will use those names as keys. If a dictionary, then the keys will be the
        keys in the output and the values are the variables in the OpenDAP repo.

        Currently I've not worked out whether OpenDAP repos can have netCDF-like groups, or how to access them if they do.

    date
        The date to download; may be omitted if that is already included in the URL.

    integrate_xgas
        Whether to integrate the VMR profile to get the Xgas value corresponding to the profile. Can be "quick", "accurate", or a boolean.
        If not ``False``, then you *must* map the VMR profile to the key "vmr" and the pressure profile to "pres" in the output 
        (using the ``variables`` keyword).

    pmax, pmin
        Only used when ``integrate_xgas = "accurate"``, these set the pressure limits to integrate between. The defaults are those used for PAN.

    Returns
    -------
    data
        A dictionary with the variables as numpy arrays in the values. 
    """
    if integrate_xgas and pmax < pmin:
        raise ValueError('pmax must be greater than pmin')

    data = ncio.read_opendap_url(url, variables, date)

    if integrate_xgas == 'quick':
        data['xgas'] = calc_xvmr_xarray(data['vmr'], data['pres'], pmax=pmax, pmin=pmin)
    elif integrate_xgas == 'accurate' or integrate_xgas is True:
        n_tgts = data['vmr'].shape[0]
        xgas = np.full(n_tgts, np.nan)
        pbar = miscutils.ProgressBar(n_tgts, prefix='Integrating Xgas')
        for i in range(n_tgts):
            pbar.print_bar()
            xgas[i], _, _ = calculate_xvmr(data['vmr'][i].data, data['pres'][i].data, pmax, pmin)

        data['xgas'] = xgas

    return data


def calc_xvmr_xarray(vmr, pres, pmax=999999.0, pmin=-0.1):
    """Estimate XVMR for xarrays of VMRs and pressure

    This is a rough approximation of the ``calculate_xvmr`` method; it doesn't have all the NaN-filling
    implemented, but instead relies on Xarray's handling of NaNs. It will be much faster on arrays of
    any significant size, however.

    The input arrays must have two dimensions, the second one must be "level", and the first one must be
    the targets.
    """
    assert len(pres.dims) == 2
    assert pres.dims[1] == 'level'
    assert vmr.dims == pres.dims

    pwf_layer = pres.isel(level=slice(None, None, -1)).diff(dim='level')
    pwf_layer = pwf_layer / pwf_layer.sum(dim='level')
    pwf_layer = pwf_layer.isel(level=slice(None, None, -1))

    pwf_level = xr.DataArray(0., dims=pres.dims, coords=pres.coords)
    pwf_level[:, :-1] = pwf_layer.data
    pwf_level[:, 1:] += pwf_layer.data
    pwf_level /= pwf_level.sum(dim='level')

    pp_outside = (pres.data < pmin - 0.05) | (pres.data > pmax + 0.05)
    pwf_level.data[pp_outside] = 0.0
    return (pwf_level * vmr).sum(dim='level')
