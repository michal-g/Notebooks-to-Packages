# Creating a package for our suite of data analyses #

![](../resources/canada.png)

Our experiment has expanded north of the border! Our Canadian colleagues need our help and we've added some features to
our analysis we'd like to share with them. What can we do to simplify the process of running and building upon the
`predicting-ufo-sightings.py` pipeline for those who we can't share the code with directly?


## An introduction to using packages in Python ##

First off: package management, which includes creating packages, is among the two or three major weaknesses of Python as
a programming language. Although the Python development team has been making progress on this front in the last few
years, it is still often a clunky and frustrating process. We have sidestepped these issues here by presenting a
packaged experiment as a fait accompli — but be aware that turning a script into a package, especially at the first time
of asking, can be a considerable investment of time, effort, and most importantly, patience.

With that said, why would we want to go through the trouble of packaging our UFO sightings experiment? As with going
from notebooks to scripts, it is only a good decision if the features you want to add to the experiment make it worth
it. Part of the art of software engineering is building an intuition for when such leaps of design should be made, for
which there are no fast and hard rules.

Nevertheless, a reasonable heuristic to use when deciding when to make a package is that it should be done when you want
to share not just your scripts, but also each of the components of your analysis (scraping the data, plotting, etc.) as
separate parts. Our final script introduced modularization, which turned these components into functions, but allowing
others to import these functions into their own code is difficult if the module they are in is not part of a package.


## Designing the structure of your package ##

In the last stage of developing our pipeline's script, we modularized the code by breaking it apart into its constituent
parts and turning each of these parts into its own function. We can now go one step further and start organizing these
functions across different files, which in Python are called _modules_. A higher-order structure prevents our code base
from becoming cluttered and unmanageable as we add more features to the experiment.

We hence have modules `data`, `plot`, and `predict`, as well as `utils`, which we'll use for helper functions used
across the other modules. How should these be arranged within the package folder structure? The package `PredUFO`
demonstrates a very simple structure for use in our case:

```
PredUFO/
+-- predufo/
    +-- __init__.py
    +-- data.py
    +-- plot.py
    +-- predict.py
    +-- utils.py
    +-- command_line.py
+-- demos/
    +-- predicting-ufo-sightings.ipynb
+-- pyproject.toml
```

There is a folder, `predufo`, which contains the source code for our analysis; we will place meta-data for the package
in the root package directory `PredUFO`. The root directory will usually contain other meta-data for the package in
addition to `pyproject.toml`, such as licensing information and READMEs, which we omit here for the sake of showing a
minimal working example.

Another common folder structure used in packages looks like this:

```
PredUFO/
+-- src/
    +-- predufo/
        +-- __init__.py
        +-- data.py
        +-- plot.py
        +-- predict.py
        +-- utils.py
        +-- command_line.py
+-- demos/
    +-- predicting-ufo-sightings.ipynb
+-- pyproject.toml
```

This is called a "src layout", as opposed to the flat layout we use above. It is
[somewhat safer](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) to create a larger
separation between the code in your package that will be imported by other packages (`from predufo.data import
scrape_sightings as scrape_ufo_sightings`), but this is unlikely to make a difference unless there are multiple
multiple developers working on your code. The flat layout generally makes more sense for packages based around command
line tools as opposed to importable Python code.

Let's consider some of the other new files we have created for our package and what they do.


### \__init\__.py and subpackages ###
We have created a file inside `predufo` named `__init__.py` which identifies this folder as a Python package. Here we
must draw the distinction between the `PredUFO` package, which is the collection of files and folders that we distribute
to others so they can run our analysis, and the `predufo` package, more narrowly defined as the collection of modules
under a specific alias that are imported into Python through `PredUFO`. A Python package can have an infinitely
recursively defined set of subpackages.

When we turn `predufo` into a Python package, its modules become available through import statements which can be used
in other places: `predufo.data.scrape_sightings()` as well as from within our package:
`from .plot import plot_totals_map, animate_totals_map`. It is common to define a list of commonly used resources in
`__init__` so that they can be imported directly: `from predufo import predict_sightings`.


### Specifying package meta-data in `pyproject.toml` ###

Much like `__init__.py` files denote that a folder is actually a Python package in the narrow sense, files like
`pyproject.toml`[^1] denote that a folder contains both a Python package's source code and its metadata, of which
`pyproject.toml` is arguably the most important part. A metadata specification file is always placed at the root of the
package, and contains fields such as `project` for general package info such as name and author contacts, `build-system`
for telling Python which backend to use for generating the package, and `project.urls` for linking to our project's
website.

The `setuptools` backend automatically recognizes our repository layout and adds the package `predufo` to the Python
namespace (this had to be specified manually in earlier distributions of Python). We can also use `project.scripts` for
creating mappings between the tools we want added to the command line namespace and their source code. The flat project
structure we chose above entails explicitly instructing Python to use the `predufo` folder as the root package.


## Sharing your package with others ##

If your package is stored in a GitHub repository, it is already being shared with others — as long as your repo is
public, anyone can download your source code. Including installation instructions in the project README (usually
centered around running `pip install .` within the cloned repository) is usually enough for interested parties to use
your package.

You can also go one step further and make your package part of a package index such as [PyPI](https://pypi.org/).
The advantage of this is cutting Git out of the installation process: users can instead instruct `pip` to download the
package from the index (basically, a repository of packages) as part of the `pip install` command. More information
about uploading a package to an index can be found in the [Python doc files](
https://packaging.python.org/en/latest/tutorials/packaging-projects/#generating-distribution-archives).


[^1]: you may have seen `setup.py` being used for package metadata specification; this configuration is now being phased
      out in newer Python versions in lieu of `pyproject.toml`
