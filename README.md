# Generative network modeling reveals a first quantitative definition of bilateral
symmetry exhibited by a whole insect brain connectome

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

## Abstract 
Comparing connectomes can help explain how neural connectivity is related to genetics, disease, development, or learning. However, making statistical inferences about the significance and nature of differences between two networks is an open problem, and such analysis has not been extensively applied to nanoscale connectomes. Here, we investigate this problem via a case study on the bilateral symmetry of a larval *Drosophila* brain connectome. We translate notions of “bilateral symmetry” to generative models of the network structure of the left and right hemispheres, allowing us to test and refine our understanding of symmetry. We find significant differences in connection probabilities both across the entire left and right networks and between specific cell types. By rescaling connection probabilities or removing certain edges based on weight, we also present adjusted definitions of bilateral symmetry exhibited by this connectome. This work shows how statistical inferences from networks can inform the study of connectomes, facilitating future comparisons of neural structures.

## Repo structure 
- ``.github``: Files specifying how the repo behaves on GitHub
- ``data``: Directory to store the raw data. 
- ``docs``: Files to build the documentation website in the form of a [Jupyter Book](https://jupyterbook.org/intro.html)
- ``pkg``: A local Python package used for analysis in the Jupyter Notebooks

## Building the book 
*Coming soon*

## Running the code
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

## Credits

This project is created using the excellent open source [Jupyter Book project](https://jupyterbook.org/) and the [executablebooks/cookiecutter-jupyter-book template](https://github.com/executablebooks/cookiecutter-jupyter-book).
