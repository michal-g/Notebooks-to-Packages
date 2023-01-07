## How to turn your Python notebooks into Python packages ##

In this workshop we will take a data analysis pipeline implemented in a Jupyter notebook and convert it into a script
that can be run from command-line. We will then convert this script into a Python package: a collection of code modules
supporting a pre-defined set of command-line tools.


### Preparing for the workshop ###

These materials are designed for users with at least some knowledge of Python, and especially using Jupyter notebooks to
build data analysis experiments. You may also want to refresh your acquiantance with the use of Python packages such as
`pandas`, `matplotlib`, and `requests` before starting this workshop.

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


### Why do we want to use packages instead of notebooks? ###

This is a very important question. The easiest answer is that, often, there is no reason to. If you are already using
Jupyter notebooks, you are familiar with how convenient they make it to create and test an experiment from scratch,
allowing you to separate different parts of the experiment across "cells" for modular execution. Especially if you want
to create plots quickly, notebooks' built-in GUI means that you can write code and produce plots within the same browser
window.

Jupyter notebooks are great for experiments that are "linear" and "one-off", meaning that they consist of a single chain
of steps carried out one after the other, and that these steps will not have to be updated or rearranged at some point
in the future. Indeed, the very visual structure of a notebook reinforces this linear nature: one cell following
another, each executed in turn. One can of course choose one's own order of executing individual cells, but this will
usually result in errors, and notebooks do not have any built-in mechanism for informing which cell depends on another —
other than the aforementioned order of the cells themselves.

This linearity simplifies things, but it is also extremely limiting in terms of the kinds of experiments we can design.
Pipelines which execute heterogenous steps in parallel are off the table, as are pipelines which reuse code from other
pipelines without simply copying the text. The modular structure of notebooks is somewhat of an illusion; in reality,
the different cells have a very rigid relationship with one another.

Jupyter notebooks are also difficult to expand upon beyond the analysis they were designed to carry out. One of the more
obvious ways this problem manifests itself is when we try to parametrize an existing experiment. If we are e.g. training
a machine learning classifier with a regularization penalty of `alpha=0.01`, and we want to try other values of alpha in
a systematic way, there is no way of doing so without manually updating the stated value of `alpha` within the notebook.
For testing a handful of values of alpha this is fine, but notebooks quickly become cumbersome if we want to test
hundreds of such values. The penalty you pay for being able to execute individual notebook cells within a pretty GUI is
the inability to turn cells (or the entire notebook) into functions with arbitrary arguments and argument values.

It is difficult to appreciate the full gravity of these considerations until one actually tries to build upon an
experiment in Jupyter. Thus an important feature of this workshop is the exposition of the intermediate steps involved
in building a script, and then a package, from a notebook. This makes it easier to study the considerations that must be
made at each step of the process, and the trade-offs inherent in engineering one's code.
