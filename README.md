# MCCE4-Tools (Alpha)
  * Tools for processing MCCE4 simulation outputs
  * A concise description is provided in `MCCE4-Tools/mcce4_tools/tools.info`:
  ```
   cd MCCE4-Tools
   cat mcce4_tools/tools.info
  ```

## Installation

### Installation Preview:
This repo is not yet published, so the "installation" is a process with these steps:
  1. Navigate to a directory of your choice
  2. Clone this repo, MCCE4-Tools:
  ```
   git clone https://github.com/GunnerLab/MCCE4-Tools.git
  ```

 3. Add the clone's path to your `.bashrc` file and save it, then source/dot your `.bashrc` file:
 ```
  # add MCCE4-Tools clone to the system path:
  export PATH="/path/to/MCCE4-Tools/mcce4_tools:$PATH"
 ```

 4. If all went well, all the command line tools are discoverable (not runable yet). You can verify their location by running the `which` command, e.g.:
 ```
  which getpdb
 ```

 5. To _run_ the tools, activate an appropriate environment associated with __python 3.10__.
   * Make sure your environment contains these __dependencies__:
   ```
    matplotlib
    mdanalysis
    networkx
    pymol-bundle
    numpy
    pandas
    parmed
    requests
    scipy
    seaborn
    'setuptools>=64'
   ```

 6. Test a tool
   * Activate your environment
   * Type `getpdb` and press Enter: the cli help should display

 7. For developing:
   * Activate your environment
   * Go into the clone directory (`cd your/path/to/cloned/MCCE4-Tools`)
   * Install the clone codebase as an editable package in your activated environment:
   ```
    pip install -e .
   ```

## Keeping your toolbase up to date:
As this is repo is bound to be frequently updated, your installation must be kept up to date.  
To do so run these commands:

  1. Go into the clone directory (`cd your/path/to/cloned/MCCE4-Tools`)
  2. Refresh the code with a `git pull`:
  ```
   git pull;
  ```
  3. Activate the appropriate environment before using any tools, e.g. mc310:
  ```
   conda activate mc310;
  ```
  4. If you have setup your clone for development, refresh the environment installation:
  ```
   pip install -e .
  ```

## Command Line tools available (as of 06-03-25):
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
