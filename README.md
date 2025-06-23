# Supporting notebook for "An algorithm to retrieve peroxyacetal nitrate from AIRS"

[![DOI](https://zenodo.org/badge/975002133.svg)](https://zenodo.org/badge/latestdoi/975002133)

## Initial setup

### Downloading this repository

- Clone the repo with the `--recurse-submodules` flag, or
- Clone the repo as normal and run `git submodule update --init --recursive` inside it afterwards.

### Downloading the data

This notebook relies on the dataset associated with the paper, available at https://doi.org/10.22002/exv89-7v481.
It must be downloaded to the directory named `data` in the same directory as this readme and the notebook.
That is, once the data are downloaded, the directory structure should look like so:

```
.
├── airs-pan-paper-figures.ipynb
├── data
│   ├── climatology_PAN_prior.nc
│   ├── goes-abi
│   ├── modis-clouds
│   ├── spectral-signatures.nc
│   ├── states.csv
│   ├── strategy_tables
│   ├── uncertainty
│   ├── validation
│   └── west_coast_fire_pca.nc
├── environment-specific.yml
├── environment.yml
├── README.md
└── src
    ├── airs-pan-ml
    ├── jllutils
    └── muses_utils
```

### Installing dependencies

This assumes that you have the [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) program available to manage Python environments.
If you use `conda` or `mamba`, subtituting those commands for `micromamba` below _should_ work.
If you use other environment management tools (`poetry`, `uv`, `pipenv`, etc.), you will need to make an environment with all the dependencies listed in `environment.yml`,
then add the custom dependencies from `src/`.

- In the top directory (i.e., the one with this README), run:
```
micromamba create -n laughner-et-al-notebook -f environment.yml
```
(You can change the name following `-n` to whatever you like, just replace that in any of the following commands.)
- If any dependencies cannot be found, try `micromamba search DEP`, e.g., `micromamba search pyhdf` and update the version in the `environment.yml` file -
conda-forge only seems to keep somewhat recent versions of some packages.
- Activate the new environment with `micromamba activate laughner-et-al-notebook`.
- `cd` into each of the directories under `src` and run `pip install -e .` in each.
- Unless you have a preferred way of creating notebook kernels, if using Jupyter Lab or the vanilla Jupyter Notebook to execute the notebook, run 
`ipython kernel install --user --name='laughner-et-al-notebook'` to create a Jupyter kernel for it.
    - If using VSCode or another way of running notebooks, follow the instructions for that program
- You will likely need to set the notebook to use this kernel at least the first time you open it.
