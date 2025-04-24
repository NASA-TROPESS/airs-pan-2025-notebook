# Supporting notebook for "An algorithm to retrieve peroxyacetal nitrate from AIRS"

## Initial setup

### Downloading this repository

- Clone the repo with the `--recurse-submodules` flag, or
- Clone the repo as normal and run `git submodule update --init --recursive` inside it afterwards.


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

