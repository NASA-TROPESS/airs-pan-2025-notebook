from pathlib import Path
import netCDF4 as ncdf
import numpy as np
import os
import pandas as pd
from typing import Optional, Union, Sequence


def convert_sounding_ids(sounding_ids: np.ndarray) -> Union[str, np.ndarray]:
    """Convert 1D or 2D arrays of sounding IDs in combined files to Unicode strings

    py-combine converts the sounding IDs to 2D integer arrays, where each element is
    one character and each row one sounding ID. This function takes such an array
    and converts it into an array of strings.

    Parameters
    ----------
    sounding_ids
        The array of sounding IDs as integers. Can be 

    Returns
    -------
    sounding_id_strings
        Decoded sounding ID(s). If the input was 1D, this is a string. If the input
        was 2D, this is an array of strings.
    """
    sounding_ids = sounding_ids.astype('uint8')
    if sounding_ids.ndim == 1:
        return sounding_ids.tobytes().decode()
    else:
        return np.array([r.tobytes().decode() for r in sounding_ids])


def convert_sounding_array(sounding_array: np.ndarray, date_array: np.ndarray) -> Union[str, np.ndarray]:
    if sounding_array.ndim != date_array.ndim:
        raise ValueError('Sounding and date arrays must have the same number of dimensions')
    if sounding_array.shape[0] != date_array.shape[0]:
        raise ValueError('Sounding and date arrays must have the same length in the first dimension')

    def make_sid(srow, drow):
        did = ''.join(str(el) for el in drow[:3])
        tid = ''.join(str(el) for el in drow[3:])
        sid = '_'.join(str(el) for el in srow)
        return f'{did}T{tid}_{sid}'


    if sounding_array.ndim == 1:
        return make_sid(sounding_array, date_array)
    else:
        output_ids = []
        for sounding_row, date_row in zip(sounding_array, date_array):
            output_ids.append(make_sid(sounding_row, date_row))
        return np.array(output_ids)


def get_nc_var(ds: ncdf.Dataset, variable: str, read_array: bool = True, fill=-999):
    """Get a netCDF variable in any arbitrary group from a netCDF file

    Parameters
    ----------
    ds
        A handle to the netCDF file to get the variable from

    variable
        Path to the variable inside the netCDF file. If the variable is in a child group,
        separate groups with a forward slash (e.g. "Characterization/Emissivity_Initital".)
        A leading slash is ignored.

    read_array
        If True, then the values in the variable are read in and returned. If False, the 
        variable itself is returned. When reading in the values, the array is converted
        to a numpy array (from a masked one). Masked elements will be replaced with ``fill``
        if the array contains integers, and ``np.nan`` otherwise. If not an integer type,
        then any values approximately equal to ``fill`` are replaced with NaNs.

    fill
        Fill value for the array. No effect if ``read_array`` is false. For floating point
        arrays, values approximately equal to this are replaced with ``np.nan``. For integer
        types, masked elements in the original array are replaced with this value.
    """
    varparts = variable.lstrip('/').split('/')
    grp = ds
    i = 0
    while i < len(varparts)-1:
        grp = grp.groups[varparts[i]]
        i += 1
    var = grp[varparts[-1]]
    if read_array:
        data = var[:]
        is_int = np.issubdtype(data.dtype, np.integer)
        data = var[:].filled(fill if is_int else np.nan)
        if fill is not None and not is_int:
            data[np.isclose(data, fill)] = np.nan
        return data
    else:
        return var


def find_var_at_path_ignore_case(ds: ncdf.Dataset, variable_path: str) -> str:
    """Find a variable path when the capitalization is not known.

    Given a variable path such as "/characterization/desert_emiss_qa", this will look for
    a variable named "desert_emiss_qa" in a group named "characterization" without considering
    the case of the characters (so "Desert_Emiss_QA" or "DESERT_EMISS_QA" would both match).
    This is useful for variables like "Longitude", "Latitude", and "SoundingId", which frequently
    have different capitalization in different files.

    Parameters
    ----------
    ds
        Open handle to the netCDF file to search

    variable_path
        The path of the variable to find, ignoring case.

    Returns
    -------
    path
        The path of the variable with correct capitalization.

    Raises
    ------
    KeyError
        If it could not find the variable, even ignoring case.

    See also
    --------
    * :func:`find_nc_var` - function to find a variable when you know the name,
      but not the group.
    """
    varparts = variable_path.lstrip('/').split('/')
    finalparts = []
    grp = ds
    i = 0
    while i < len(varparts)-1:
        # Until we're at the last part of the variable path, we're looking for a group.
        group_target = varparts[i].lower()
        found_group = False
        for ds_grpname in grp.groups.keys():
            ds_grpname_lower = ds_grpname.lower()
            if ds_grpname_lower == group_target:
                grp = ds.groups[ds_grpname]
                finalparts.append(ds_grpname)
                i += 1
                found_group = True
                break

        if not found_group:
            missing_path = '/' + '/'.join(varparts[:i+1])
            raise KeyError(f'Unable to find a case insensitive match for {missing_path} in {ds.filepath()}')

    # Now we look for the variable
    var_target = varparts[-1].lower()
    found_variable = False
    for ds_varname in grp.variables.keys():
        ds_varname_lower = ds_varname.lower()
        if ds_varname_lower == var_target:
            finalparts.append(ds_varname)
            found_variable = True
            break

    if not found_variable:
        raise KeyError(f'Unable to find a case insensitive match for {variable_path} in {ds.filepath()}')

    return '/' + '/'.join(finalparts)


def find_nc_var(ds: ncdf.Dataset, variable: str, ignore_case: bool = True, return_type: str = 'path', fill=-999) -> Union[str, ncdf.Variable, np.ndarray]:
    """Find a variable in a netCDF file by its name alone, searching all groups recursively.

    In contrast to :func:`find_var_at_path_ignore_case` (which expects you know exactly where the variable is in the file, just
    not how it is capitalized), this function can search for a variable which may be in any group of a netCDF file with only
    the name of the variable itself.

    Parameters
    ----------
    ds
        A handle to the open netCDF dataset to search.

    variable
        The name of the variable to look for.

    ignore_case
        Set to ``False`` to respect the capitalization of the variable name when searching.

    return_type
        Determines what is returned:

        * "path" (default) returns the path to the variable, suitable to use in a call to :func:`get_nc_var` or one of the plot reader classes.
        * "var" or "variable" returns the :class:`netCDF4.Variable` instance for the specified variable
        * "array" returns the data array for the variable as read by :func:`get_nc_var`

    fill
        Same purpose as in :func:`get_nc_var`, only used if ``return_type="array"``

    Returns
    -------
    variable
        The variable path, instance, or data depending on the value of ``return_type``. Note that the *first* instance of a variable
        with the given name is returned. If multiple variables in different groups have the same name, whether the same one is always
        returned depends on whether your :mod:`netCDF4` package iterates through groups in a deterministic manner.

    Raises
    ------
    KeyError
        If it cannot find any variable with the given name.

    See also
    --------
    * :func:`find_var_at_path_ignore_case` - to use when you know the path of the variable, but it can have various capitalizations.
    """
    def get_varname(varlist):
        if ignore_case:
            test = variable.lower()
            for v in varlist:
                if test == v.lower():
                    return v
        elif variable in varlist:
            return variable
        else:
            return None

    def find_inner(grp, parts=None):
        if parts is None:
            parts = ['']
        varkey = get_varname(grp.variables.keys())

        if varkey is not None:
            parts.append(varkey)
            return '/'.join(parts)

        for grpname, grp2 in grp.groups.items():
            var = find_inner(grp2, parts + [grpname])
            if var is not None:
                return var

        return None

    var = find_inner(ds)
    if var is None:
        raise KeyError(f'Could not find a variable named "{variable}" in any group of this file.')

    if return_type == 'path':
        return var
    elif return_type in {'var', 'variable'}:
        read_array = False
    elif return_type in {'array'}:
        read_array = True
    else:
        raise TypeError(f'return_type must be one of: "path", "var", "variable", or "array", not "{return_type}"')

    return get_nc_var(ds, var, read_array=read_array, fill=fill)


def read_quality_file(file, master_only: bool = False) -> pd.DataFrame:
    """Read a quality flags file.

    This reads in a file from the ``QualityFlags`` directory in the OSP strategy tables and
    returns the flags table as a dataframe.

    Parameters
    ----------
    file
        Path to the file to read

    master_only
        Set to ``True`` to only keep rows in the dataframe when that flag will be 
        used for the master quality flag.

    Returns
    -------
    dataframe
        A dataframe with the quality flag information. This usually is the name of the flagging variable,
        the min and max cutoffs, and whether it is used for the master quality flag.

    See also
    --------
    * :func:`read_quality_file_for_species` - a convenience function if your OSPs are installed in the standard location.
    """

    with open(file) as f:
        for line in f:
            if line.lower().startswith('end_of_header'):
                break
        df = pd.read_csv(f, sep=r'\s+', na_values='na').dropna()
    df['Use_For_Master'] = df['Use_For_Master'].astype('int')
    df.set_index('Flag', inplace=True)
    if master_only:
        xx = df['Use_For_Master'] == 1
        return df[xx]
    else:
        return df


def read_quality_file_for_species(strategy_table: str, species: Optional[str], user: Optional[str] = None, master_only: bool = False, strat_table_dir=None, hide_path: bool = False) -> pd.DataFrame:
    """Read a quality file for a given species from a strategy table, assuming the standard OSP layout.

    This will read the file at "~/OSP/Strategy_Tables/$USER/$STRATEGY_TABLE/QualityFlags/QualityFlag_Spec_Nadir_$SPECIES.asc".
    As long as your OSPs are laid out like this, this can be a more convenient way to read the quality flag file for a given
    strategy/species. If not, you'll need to use :func:`read_quality_file` to give the file path directly.

    Parameters
    ----------
    strategy_table
        The name of the strategy table to read from, e.g. "OSP-CrIS-v10".

    species
        The name of the specie or species in the quality flag file name, e.g. "PAN" or "H2O_O3". Pass ``None`` to read the base 
        "QualityFlag_Spec_Nadir.asc" file.

    user
        Since users are expected to make a subdirectory of their own under the "Strategy_Tables" directory if they want to make
        custom tables, the default behavior is to look for a subdirectory there with the same name as the login for the current
        user. If found, that is where this function looks for the quality flag files. If not found, then it uses the "ops" directory.
        You can also pass a specific directory name as this parameter to override that logic.

    master_only
        Set to ``True`` to only keep rows in the dataframe when that flag will be 
        used for the master quality flag.

    strat_table_dir
        If given, this must be the path to the directory containing the various strategy tables; ``user`` will be ignored.

    hide_path
        By default, this function will print the file path to the table it read, as a way for you to check that it's reading the
        one you expected. You can turn that off by setting this argument to ``True``.

    Returns
    -------
    dataframe
        A dataframe with the quality flag information. This usually is the name of the flagging variable,
        the min and max cutoffs, and whether it is used for the master quality flag.

    See also
    --------
    * :func:`read_quality_file` - read a quality flag file given directly by its path.
    """
    if strat_table_dir is None:
        strat_table_dir = Path('~/OSP/Strategy_Tables').expanduser()

        if not strat_table_dir.exists():
            raise IOError('You do not appear to have the OSPs linked to ~/OSP or there is not a Strategy_Tables directory in there.')

        if user is None:
            user = os.getlogin()
            if not (strat_table_dir / user).exists():
                user = 'ops'

        strat_table_dir = strat_table_dir / user 
        if not strat_table_dir.exists():
            raise IOError(f'Cannot find OSPs for user "{user}"')
    else:
        strat_table_dir = Path(strat_table_dir)

    strat_table_dir = strat_table_dir / strategy_table
    if not strat_table_dir.exists():
        raise IOError(f'Cannot find strategy table {strategy_table}')

    if species is None:
        quality_flag_file = strat_table_dir / 'QualityFlags' / 'QualityFlag_Spec_Nadir.asc'
    else:
        quality_flag_file = strat_table_dir / 'QualityFlags' / f'QualityFlag_Spec_Nadir_{species}.asc'
    if not quality_flag_file.exists():
        raise IOError(f'Cannot find a "QualityFlag_Spec_Nadir*.asc" file for species {species}')

    if not hide_path:
        print(f'Reading quality flag file {quality_flag_file}')

    return read_quality_file(quality_flag_file, master_only=master_only)



def read_many(targets_dir: str, product_file: str, variables: Sequence[str]) -> dict:
    """A function to read from all targets' product files in a setup-targets directory

    This is meant for use when py-combine cannot be used or isn't necessary for quick checks.

    Parameters
    ----------
    targets_dir
        Path to the setup-targets or equivalent directory. All sub-directories starting with "20"
        in this directory will be searched as targets.

    product_file
        The name of the output file (basename only) to load from in each target.

    variables
        The sequence of variables to load; if a variable is in a subgroup, include the group as part
        of the path e.g. "Retrieval/Column".

    Returns
    -------
    dict
        A dictionary with ``variables`` as its keys and the concatenated data from the targets as values.
        The data arrays will have the targets dimensions as their first. Float arrays use NaNs for fill values,
        integer arrays use -999.
    """
    targets = sorted(Path(targets_dir).glob('20*'))
    ntgt = len(targets)

    # Get the shapes and types of all the variables first so we can create
    # arrays of the right size with fill values populated
    data = dict()
    fills = dict()
    first_product_file = None
    for target in targets:
        this_product_file = target / 'Products' / product_file
        if this_product_file.exists():
            first_product_file = this_product_file
            break

    if first_product_file is None:
        raise IOError(f'Could not find product file "{product_file}" in any of the {ntgt} directories. Either the file name is wrong, or all soundings failed')

    with ncdf.Dataset(first_product_file) as ds:
        for var in variables:
            dtype = ds[var].dtype
            if np.issubdtype(dtype, np.floating):
                fill = np.nan
            elif np.issubdtype(dtype, np.integer):
                fill = -999
            else:
                raise NotImplementedError(f'No fill value defined for data type {dtype}')

            shape = ds[var].shape
            data[var] = np.full((ntgt,) + shape, fill)
            fills[var] = fill

    for itgt, target in enumerate(targets):
        this_product_file = target / 'Products' / product_file
        if not this_product_file.exists():
            continue

        with ncdf.Dataset(this_product_file) as ds:
            for var in variables:
                data[var][itgt] = ds[var][:].filled(fills[var])

    return data
