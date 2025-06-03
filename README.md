# MCCE4-Tools
 * Tools for processing MCCE4 simulation outputs
 * A concise description is provided in `MCCE4-Tools/tools.info`.

## Installation
This repo is not yet published, so the installation is done via [`unidep`](https://unidep.readthedocs.io/en/latest/faq.html#q-when-to-use-unidep) to obtain an editable codebase.

 1. _Optional:_ Navigate to a specific directory
 2. Clone this repo, MCCE4-Tools:
 ```
 git clone https://github.com/GunnerLab/MCCE4-Tools.git
 ```

 3. Activate an environment associated with python 3.10 that includes `setuptools` (version 64+) and `unidep`, or create one:
 ```
 conda create -n mc310 python=3.10 'setuptools>=64' unidep ;
 conda activate mc310
 ```

### Important note:
When you install the codebase & it's cli tools in the next step, the `pymol` package will be installed in the activated environment, which you may not want to do if you already have an installation in some other location that has been referenced in you PATH variable.  
In this case, you need to comment out the two lines in the `MCCE4-Tools/pyproject.toml` file that are preceeded by this informational comment line:
```
# comment out this next line if you do not want pymol installed:
```
then save and close the file before proceeding.

 4. Install the repo codebase as an editable package in your environment:
 ```
  unidep install -e ./MCCE4-Tools
 ```

 5. If all went well, the installation script has installed all the command line tools defined in `MCCE4-Tools/pyproject.toml` in your environment. You can verify their location by running the `which` command, e.g.:
 ```
 which cif2pdb
 ``` 

## Line commands created in your environment (as of 06-03-25):
 * `cif2pdb`
 * `clear_mcce_folder`
 * `getpdb`
 * `ms_protonation`
 * `pdbs2pse`
 * `postrun`

## The `notebooks` folder
This folder contain examples of usage of the codebase instead of, or in addition to, the command-line tools.  

Currently, the [`ms_protonation_analysis` notebook](./notebooks/ms_protonation_analysis.ipynb) is a walk-through of the processing pipeline used in the `ms_protonation` tool.  
As this tool _requires_ a parameter file at the command line; two files with a '.crgms' extension are provided as examples.

## Test (using the same activated environment): `$ cmd +ENTER`
 1. Type `cif2pdb` and press Enter: the cli help should display
 2. Type `getpdb` and press Enter: the cli help should display

## Keeping your toolbase up to date:
As this is repo is bound to be frequently updated, your installation must be kept up to date.  
To do so run these commands:

1. Go to the directory where you cloned the MCCE4-Tools repo;
2. Enter the clone & refresh the code with a `git pull`:
 ```
  cd MCCE4-Tools;
  git pull;
 ```
3. Activate the appropriate environment, e.g. mc310:
 ```
  conda activate mc310;
 ```
4. Refresh the environment installation:
 ```
  unidep install -e .
 ```
