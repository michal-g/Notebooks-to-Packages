## Creating a package for our suite of data analyses ##

![](../resources/calling.jpg)

Our experiment is now fleshed out, and we would like to share it with other researchers for their own use. What can we
do to simplify the process of running and building upon the experiment in `predicting-ufo-sightings.py` for those who we
can't give or explain the code to directly?


### An introduction to using packages in Python ###

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


### Designing the structure of your package ###


### The `pyproject.toml` file ###

When creating a Python package we can use a specification file, always named `pyproject.toml` and placed at the root of
the package, to specify which parts of our code are designed to be used in other analyses as well. We can also create
programs that can be run from command line; in effect these are equivalent to scripts which have been given handy
aliases (e.g. `predUSA` for `predicting-ufo-sightings.py`).


### Sharing your package with others ###
