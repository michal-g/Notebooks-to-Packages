## Turning your Python notebooks into Python packages ##

In this workshop we will take a data analysis pipeline implemented in a Jupyter notebook and convert it into a script
that can be run from command-line. We will then convert this script into a Python package: a collection of code modules
supporting a pre-defined set of command-line tools.

The purpose of this course is **not** to dissuade you from using Jupyter! Notebooks are an incredibly accessible and
powerful tool for data scientists and researchers alike. However, as an experiment expands in scope and scale, the
limiting features of notebooks start to become readily apparent. We will focus on the _process_ of software design:
where and when in the course of building an analysis pipeline you may want to consider investing the effort to leverage
the other tools at your disposal as a Python developer.


## Table of contents ##

1. **Notebooks**  
    - introducing a data analysis implemented in a Jupyter notebook
    - advantages and disadvantages of notebooks
    - when to consider moving beyond a notebook


2. **Scripts**  
    - converting a notebook into a script  
    - parametrizing a script  
    - modularizing a script


3. **Packages**
    - converting a script into a package
    - applying object-oriented programming within a package


### Preparing for the workshop ###

These materials are designed for users with at least some knowledge of Python, and particuarly with using Jupyter
notebooks to build data analysis experiments. You may also want to refresh your acquiantance with the use of Python
packages such as `requests`, `pandas`, `matplotlib`, and `scikit-learn` before starting this workshop.

To run the code included in this workshop, you'll need access to a command-line environment with a
[conda](https://conda.io/projects/conda/en/latest/index.html) installation. In this environment, choose a place to check
out the course repository:

```git clone git@github.com:michal-g/Notebooks-to-Packages.git```

In the newly-created folder `Notebooks-to-Packages` you'll find a file specifying a conda environment; we create the
environment and activate it using:

```
cd Notebooks-to-Packages
conda env create -f environment.yml
conda activate notebooks-packages
```
