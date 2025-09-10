# MCCE4-Tools (Alpha)
  * Tools for processing MCCE4 simulation outputs.
  * A concise description is provided in the file `tools.info` in `MCCE4-Tools/mcce4_tools`. If you are reading this document online, you can view this file [here](https://raw.githubusercontent.com/GunnerLab/MCCE4-Tools/refs/heads/main/mcce4_tools/tools.info).

## Installation Steps:
  1. Navigate to a directory of your choice (referred to as 'clone_dir'):
  2. Clone this repo, MCCE4-Tools:
  ```
   git clone https://github.com/GunnerLab/MCCE4-Tools.git
  ```

  3. Add the clone's path to your `.bashrc` (`.bash_profile`) file, save it, then "dot" or source the file:
  ```
   # add MCCE4-Tools clone to the system path:
   export PATH="clone_dir/MCCE4-Tools/mcce4_tools:$PATH"
  ```

  4. If all went well, all the command line tools are discoverable (but _not runable yet_). You can verify their location by running the `which` command, e.g.:
  ```
   which getpdb
  ```

  5. To _run_ the tools, either:
    - Activate an appropriate environment associated with __python 3.10__ (and that includes the packages in the environment file, `MCCE4-Tools/mct4.yml`).
    - Alternatively, you can create one with the provided `MCCE4-Tools/mct4.yml` file:
      * Choose one of these two options to create the environment:
        1. Option 1: To use the default environment name of 'mct4':
        ```
         conda env create -f mct4.yml
        ```
        2. Option 2: If you want something else, e.g. 'new_env' to be the environment name instead of 'mct4':
        ```
         conda env create -f mct4.yml -n new_env
        ```
      * __NOTE__
      Although pymol is necessary for certain tools, it is not included in `mct4.yml` due to licensing; installation details for PyMOL 3.1 (Version 3.1.6.1) is [here](https://www.pymol.org/)

  6. Test a tool
    * Activate your environment, e.g. `conda activate mct4`
    * Type `getpdb` and press Enter: the cli usage should display

<!--- Section commented out:
      Would likely conflict with the 'bashrc setup' described above, which also enables
      pogrammatic access.

  7. For developing:
    * Activate your environment, e.g. `conda activate mct4`
    * Go into the clone directory (`cd MCCE4-Tools`)
    * Install the clone codebase as an editable package in your activated environment:
    ```
     pip install -e .
    ```
    * Test the package:
    Open the Python interpreter, then test an import statement from the package:
    ```
     from mcce4_tools.mcce4 import constants
     print(dir(constants))

     # OR
     import mcce4_tools.mcce4 as mct
     dir(mct.constants)
    ```
-->

## Keeping your toolbase up to date:
As this is repo is bound to be frequently updated, your installation must be kept up to date.  
To do so run these commands:

  1. Go into the clone directory (`cd clone_dir/MCCE4-Tools`)
  2. Refresh the code with a `git pull`:
  ```
   git pull;
  ```

<!--- Section commented out: not absolutely necessary for current setup (i.e. refs in .bashrc).
  3. If you have setup your clone for development, refresh the environment installation:
  ```
   conda activate mct4;
   pip install -e .
  ```
-->

## Command line tools descriptions:
  * A concise description is viewable from your clone's `tools.info` file:
  ``` 
   cat MCCE4-Tools/mcce4_tools/tools.info
  ```

## The `notebooks` folder
This folder contain examples of usage of the codebase instead of, or in addition to, the command-line tools.  

Currently, the [`ms_protonation_analysis` notebook](./notebooks/ms_protonation_analysis.ipynb) is a walk-through of the processing pipeline used in the `ms_protonation` tool.  
As this tool _requires_ a parameter file at the command line; two files with a '.crgms' extension are provided as examples.
