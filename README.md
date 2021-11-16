# Maggot brain, mirror image? A statistical analysis of bilateral symmetry in an insect brain connectome

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

## Abstract 
Neuroscientists have many questions about connectomes that revolve around the ability to compare networks. For example, comparing connectomes could help explain how neural wiring is related to individual differences, genetics, disease, development, or learning. One such question is that of bilateral symmetry: are the left and right sides of a connectome the same? Here, we investigate the bilateral symmetry of a recently presented connectome of an insect brain, the Drosophila larva. We approach this question from the perspective of two-sample testing for networks. First, we show how this question of “sameness” can be framed as a variety of different statistical hypotheses, each with different assumptions. Then, we describe test procedures for each of these hypotheses. We show how these different test procedures perform on both the observed connectome as well as a suite of synthetic perturbations to the connectome. We also point out that these tests require careful attention to parameter alignment and differences in network density in order to provide biologically-meaningful results. Taken together, these results provide the first statistical characterization of bilateral symmetry for an entire brain at the single-neuron level, while also giving practical recommendations for future comparisons of connectome networks.

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
