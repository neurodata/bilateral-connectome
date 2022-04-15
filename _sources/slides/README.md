# Slides
To add your slides to the JupyterBook documentation, make sure you: 
- Make a folder with the name of your talk/poster.
- Create whatever final output you want to show as a `.pdf` file with the same name as the folder.
- Add the name of the folder to `manifest.txt` in this folder.
- Add the name of the `.pdf` as a URL to `docs/_toc.yml`. An example looks like:
  ```
  - caption: Slides
    chapters:
    - url: https://docs.neurodata.io/bilateral-connectome/drexel.pdf
      title: Drexel
  ```
