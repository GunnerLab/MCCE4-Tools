# MCCE4-Tools (Alpha)
 * Tools for processing MCCE4 simulation outputs
 * A concise description is provided in `MCCE4-Tools/tools.info`.

## Installation

### Installation Preview:
This repo is not yet published, so the "installation" is a process with two main steps:
 1. For tool use only:
   * Clone the repo
   * Add its paths to your `.bashrc` file
 2. For development:
   * After step 1, do a local software installation with `pip` to obtain an editable codebase

### Installation Detailed Steps:
 1. Navigate to a directory of your choice
 2. Clone this repo, MCCE4-Tools:
 ```
 git clone https://github.com/GunnerLab/MCCE4-Tools.git
 ```

 3. Add these clone paths to your `.bashrc` file:
 ```
 # add MCCE4-Tools clone to path:
 export PATH="/path/to/MCCE4-Tools/mcce4_tools:$PATH"
 export PATH="/path/to/MCCE4-Tools/mcce4_tools/bin:$PATH"
 ```

 * Then source/dot your `.bashrc` file.

 4. If all went well, all the command line tools are available. You can verify their location by running the `which` command, e.g.:
 ```
 which cif2pdb
 ```

 5. To run the tools, activate an appropriate environment associated with __python 3.10__.
  * Make sure your environment contains these __dependencies__:
  ```
   matplotlib
   mdanalysis
   pymol-bundle
   numpy
   pandas
   parmed
   requests
   scipy
   seaborn
   'setuptools>=64'
  ```

 5. Test a tool
  * Activate your environment
  * Type `getpdb` and press Enter: the cli help should display
  
 6. For development:
  * Activate your environment
  * Install the clone codebase as an editable package in your activated environment:
  ```
   pip install -e ./MCCE4-Tools
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
  pip install -e .
 ```
