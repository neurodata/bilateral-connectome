# Generative network modeling reveals a first quantitative definition of bilateral symmetry exhibited by a whole insect brain connectome

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

## Abstract 
Comparing connectomes can help explain how neural connectivity is related to genetics, disease, development, or learning. However, making statistical inferences about the significance and nature of differences between two networks is an open problem, and such analysis has not been extensively applied to nanoscale connectomes. Here, we investigate this problem via a case study on the bilateral symmetry of a larval *Drosophila* brain connectome. We translate notions of “bilateral symmetry” to generative models of the network structure of the left and right hemispheres, allowing us to test and refine our understanding of symmetry. We find significant differences in connection probabilities both across the entire left and right networks and between specific cell types. By rescaling connection probabilities or removing certain edges based on weight, we also present adjusted definitions of bilateral symmetry exhibited by this connectome. This work shows how statistical inferences from networks can inform the study of connectomes, facilitating future comparisons of neural structures.

## Repo structure 
- ``.github``: Files specifying how the repo behaves on GitHub.
- ``data``: Directory to store the raw data. 
- ``docs``: Files to build the documentation website (in the form of a [Jupyter Book](https://jupyterbook.org/intro.html)), talks, and posters.
- ``overleaf``: Link to an Overleaf document as a git submodule.
- ``pkg``: A local Python package used for analysis in the Jupyter Notebooks/Python scripts.
- ``results``: Place to store intermediate outputs, figures, and saved variables. 
- ``sandbox``: Junk scripts not part of the final paper/project.
- ``scripts``: Python scripts used to do all analyses
- ``shell``: Shell scripts used to run the entire project, copy results, etc.

## Getting the code and setting up an environment
Prerequisites: `git`, working knowledge of Python and command line tools.

### Using Poetry
I recommend using [Poetry](https://python-poetry.org/) to create and manage a 
reproducible environment for running the code for this project. 
- If you don't have it already, [install Poetry](https://python-poetry.org/docs/#installation) following their linked instructions.
- Navigate to a directory where you want to store the project, and clone this repo: 
   ```
   git clone https://github.com/neurodata/bilateral-connectome
   ```
- (TEMPORARY) Clone the sister repository, `giskard`:
  ```
  git clone https://github.com/bdpedigo/giskard.git
  ```
  - Note: once the code is stable, this will be replaced by an install from PyPI
- Enter the newly cloned directory:
  ```
  cd bilateral-connectome
  ```
- Create a Poetry environment:
  ```
  poetry env use python3.9
  ```
  - For me, the output looks like
     ```
    Creating virtualenv bilateral-connectome in /Users/bpedigo/bilateral-test/bilateral-connectome/.venv
    Using virtualenv: /Users/bpedigo/bilateral-test/bilateral-connectome/.venv
    ```
  - Note: this requires that you have a python3.9 installation on your machine. It is
    possible that the code for this project will run with other versions of Python,
    but I haven't tested it.
- To activate the new environment, do 
  ```
  source .venv/bin/activate
  ```
  - If you need to deactivate this environment, just type `deactivate`

### Using `pip`
*Coming soon*

## Getting the data 
*Coming soon*

## Running the code
- Make sure your virtual environment from the last section is active.
- Now you should be able to run any individual python files like normal, for example: 
  ```
  python ./bilateral-connectome/scripts/er_unmatched_test.py
  ```
- Instead of running as a Python file, you can also easily convert a Python file to a
  notebook, execute it, and have it included in the documentation folder for rendering
  as a JupyterBook. To do so, use the `make_notebook.sh` script and pass in the name of 
  the python file (without the `.py` file extension):
  ```
  sh ./bilateral-connectome/shell/make_notebook er_unmatched_test
  ```
  If you'd like to build that individual notebook and then rebuild the documentation,
  just add the `-b` argument to the same script:
  ```
  sh ./bilateral-connectome/shell/make_notebook -b er_unmatched_test
  ```
- You can also build and run all notebooks which are essential to the final paper via
  the `make_project.sh` script: 
  ```
  sh ./bilateral-connectome/shell/make_project.sh
  ```

## Building the book 
*Coming soon*

<!-- ## Usage

### Building the book

If you'd like to develop on and build the Maggot connectome book, you should:

- Clone this repository and run
- Run `pip install -r requirements.txt` (it is recommended you do this within a virtual environment)
- (Recommended) Remove the existing `Maggot connectome/_build/` directory
- Run `jupyter-book build Maggot connectome/`

A fully-rendered HTML version of the book will be built in `Maggot connectome/_build/html/`.

### Hosting the book

The html version of the book is hosted on the `gh-pages` branch of this repo. A GitHub actions workflow has been created that automatically builds and pushes the book to this branch on a push or pull request to main.

If you wish to disable this automation, you may remove the GitHub actions workflow and build the book manually by:

- Navigating to your local build; and running,
- `ghp-import -n -p -f Maggot connectome/_build/html`

This will automatically push your build to the `gh-pages` branch. More information on this hosting process can be found [here](https://jupyterbook.org/publish/gh-pages.html#manually-host-your-book-with-github-pages).

-->

<!-- ## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book). -->
