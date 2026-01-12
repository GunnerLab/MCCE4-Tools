# MCCE4-Tools (alpha)
  * Tools for processing MCCE4 simulation outputs.
  * A concise description is provided in the file `tools.info` in `MCCE4-Tools/mcce4_tools`. If you are reading this document online, you can view this file [here](https://raw.githubusercontent.com/GunnerLab/MCCE4-Tools/refs/heads/main/mcce4_tools/tools.info).

## Installation Steps:
  1. Navigate to a directory of your choice (referred to as 'clone_dir'):
  2. Clone this repo, MCCE4-Tools & cd into it (cpy and paste this command into your terminal):
  ```bash
   git clone https://github.com/GunnerLab/MCCE4-Tools.git; cd MCCE4-Tools;
  ```

  3. Add the clone's path to your `.bashrc` (`.bash_profile`) file, save it, then "dot" or __source the file__:
  ```bash
   # CHANGE 'clone_dir' to your path!

   export PATH="clone_dir/MCCE4-Tools/mcce4_tools:$PATH"
  ```

  4. If all went well, all the command line tools are discoverable (but _not runable yet_). You can verify their location by running the `which` command, e.g.:
  ```bash
   which getpdb
  ```

  5. To _run_ the tools, either:
    - Activate an appropriate environment associated with __python 3.10__ (and that includes the packages in the environment file, `MCCE4-Tools/mct4.yml`).
    - Alternatively, you can create one with the provided `MCCE4-Tools/mct4.yml` file:
      * Choose one of these two options to create the environment:
        1. Option 1: Create a NEW environment named 'mc4' if you do not have one already. To find out, run the command: `conda env list`. You do not have an 'mc4' environment if it's not listed in its output:
        ```bash
         conda env create -f mct4.yml
        ```
        2. Option 2: Update your existing 'mc4' environment (which you have already created if you have installed MCCE4-Alpha):
        ```bash
         conda env update -n mc4 -f mct4.yml
        ```
  6. Test a tool
    * Activate your environment, e.g. `conda activate mc4`
    * Type `getpdb` and press Enter: the cli usage should display

  * __NOTES__
     - Although pymol is necessary for certain tools, it is not included in `mct4.yml` due to licensing; installation details for PyMOL 3.1 (Version 3.1.6.1) is [here](https://www.pymol.org/)

  7. Setup to access the codebase programmatically:
    * Activate your environment, e.g. `conda activate mc4`
    * Go into the clone directory (`cd MCCE4-Tools`)
    * Install the clone codebase as an editable package in your activated environment:
    ```bash
     pip install -e .
    ```
    * Test the package:
    Open the Python interpreter, then test an import statement from the package:
    ```python
     from mcce4_tools.mcce4 import constants
     print(dir(constants))

     # OR
     import mcce4_tools.mcce4 as mct
     dir(mct.constants)
    ```

## Keeping your toolbase up-to-date:
As this is repo is frequently updated, your installation must be kept up to date. 
Please, run the following commands __before__ using any of its command line tools or accessing its codebase programmatically:
```bash
here=$(pwd);
clone=$(dirname $(dirname "$(python3 -c "import os, sys; print(os.path.realpath(sys.argv[1]))" "$(which ms_protonation)")"));
cd "$clone";
git pull;
cd "$here";
```

## Command line tools descriptions:
  * A concise description is viewable from your clone's `tools.info` file:
  ```bash 
   cat MCCE4-Tools/mcce4_tools/tools.info
  ```

## The `notebooks` folder
This folder contain examples of usage of the codebase instead of, or in addition to, the command-line tools.  

Currently, the [`ms_protonation_analysis` notebook](./notebooks/ms_protonation_analysis.ipynb) is a walk-through of the processing pipeline used in the `ms_protonation` tool.  
As this tool _requires_ a parameter file at the command line; two files with a '.crgms' extension are provided as examples.
