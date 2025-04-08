from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import numpy as np
import xarray as xr

class ImageProvider(ABC):
    """Base class that defines an interface to add various background images to cartopy maps.
    """
    @abstractmethod
    def add_image(self, ax):
        """Add the background image to the Matplotlib axes ``ax``.
        """
        pass
    
class StockImageProvider(ImageProvider):
    """Image provider that adds the stock Cartopy image to the axes.
    """
    def add_image(self, ax):
        ax.stock_img()
        
class GoesImageProvider(ImageProvider):
    """Add GOES truecolor imagery from an ABI L2 MCMIPC file to a map.

    GOES imagery can be downloaded from various places, but the best I've found (because they do not
    need to be ordered) is to use the Google Cloud buckets at 
    https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01502.

    The usual way to instantiate this class would be to use the :meth:`from_abi_file` class method,
    which just needs a path to the L2 ABI file. However, if you have the GOES RGB information and
    map projection information already loaded, you may pass those directly to the init method.

    Parameters
    ----------
    goes_rgb
        A 3D array of RGB values ([:,:,0] = red, [:,:,1] = green, [:,:,2] = blue) normalized to 0 to 1.

    goes_proj
        The projection that the GOES data are defined on.

    goes_x, goes_y
        The x & y coordinates of the GOES data in the projection defined by ``goes_proj``.
    """
    def __init__(self, goes_rgb: np.ndarray, goes_proj: ccrs.Projection, goes_x: np.ndarray, goes_y: np.ndarray):
        self._rgb = goes_rgb
        self._proj = goes_proj
        self._x = goes_x
        self._y = goes_y
        
    @classmethod
    def from_abi_file(cls, abi_file):
        """Create an instance of this class from an L2 ABI MCMIPC file.
        """
        rgb = cls.load_goes_rgb(abi_file)
        proj, x, y = cls.load_goes_proj(abi_file)
        return cls(rgb, proj, x, y)
    
    def add_image(self, ax):
        ax.imshow(self._rgb, origin='upper', extent=(self._x.min(), self._x.max(), self._y.min(), self._y.max()), transform=self._proj)

    @staticmethod
    def load_goes_rgb(goes_abi_file, gamma=2.2):
        """Load RGB data from a GOES MCMIPC file using true green and the given gamma correction.

        Code adapted from https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html.
        """
        with ncdf.Dataset(goes_abi_file) as ds:
            red = ds['CMI_C02'][:].filled(np.nan)
            red = np.power(np.clip(red, 0, 1), 1/gamma)
            
            green = ds['CMI_C03'][:].filled(np.nan)
            green = np.power(np.clip(green, 0, 1), 1/gamma)
            
            blue = ds['CMI_C01'][:].filled(np.nan)
            blue = np.power(np.clip(blue, 0, 1), 1/gamma)
            
            green_true = 0.45 * red + 0.1 * green + 0.45 * blue
            green_true = np.clip(green_true, 0, 1)  # apply limits again, just in case.
            
        return np.dstack([red, green_true, blue])

    @staticmethod
    def load_goes_proj(goes_abi_file):
        """Load the GOES map projection and x/y coordinate information.
        
        Adapted from https://unidata.github.io/python-gallery/examples/mapping_GOES16_TrueColor.html."""
        with xr.open_dataset(goes_abi_file) as ds:
            dat = ds.metpy.parse_cf('CMI_C02')
            proj = dat.metpy.cartopy_crs
            return proj, dat.x, dat.y



class GeoTiffImageProvider(ImageProvider):
    def __init__(self, geotiff: Union[str, Path]) -> None:
        self._geotiff = geotiff

    def add_image(self, ax):
        self.plot_geotiff(self._geotiff, ax=ax)

    @staticmethod
    def plot_geotiff(geotiff: Union[str, Path], ax=None):
        """Plot RGB data from a GeoTIFF file

        This requires that :mod:`rasterio` be installed for xarray to read the GeoTIFF.

        .. note::
            This has only been tested on GeoTIFFs generated with GDAL of MODIS Aqua data as of 2022-12-23.
            It's possible that other images might have incorrect color or orientation.

        Parameters
        ----------
        geotiff 
            Path to the GeoTIFF file to plot.

        ax
            Optional set of matplotlib axes to draw the image in. If given, must have a Cartopy projection.
            If not given, ones with the Plate Carree projection are created.

        Returns
        -------
        ax
            The axes drawn in, either those passed in or the ones created.
        """
        with xr.open_rasterio(geotiff) as gt:
            img = np.moveaxis(gt.data, 0, 2)
            extent = [gt.x.min().item(), gt.x.max().item(), gt.y.min().item(), gt.y.max().item()]
            if ax is None:
                _, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
            ax.imshow(img, origin='upper', extent=extent, transform=ccrs.PlateCarree())
            return ax