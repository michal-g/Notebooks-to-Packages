# Analyzing a public dataset using a Jupyter notebook #

![](../resources/mulder-scully.jpeg)


We found a [website](https://nuforc.org/databank/) listing reported sightings of unidentified flying objects (UFOs)
and created a notebook implementing a machine learning algorithm that predicts numbers of sightings in a set of years.
In this chapter we will examine why notebooks are useful tools for researchers and data scientists, and consider when
and why they become limiting as an analysis pipeline grows in scope.

If you have a Google account, use the above link to run the UFO sightings notebook in Colab. Otherwise, either download
the notebook and run it with the version of Jupyter installed as part of this repository's `notebooks-packages` conda
environment, or use Princeton's
[compute services](https://researchcomputing.princeton.edu/support/knowledge-base/jupyter) to run the notebook remotely.


## Notebooks: the good side ##

Let's consider why a notebook was so handy for our first step in this analysis. The primary advantage of notebooks is
that they make it very easy to get started doing data science, especially for users without a lot of software
engineering experience. The key features to consider are _interactivity_, _modularity_, and _portability_.

Notebooks are designed to be run through a graphic user interface (GUI). This is the most obvious attribute of a Jupyter
notebook: users can arrange and run code within a browser window using cells and buttons that are manipulated directly
rather than through a command-line interface. In the usual notebook workflow we alternate between entering code directly
into a cell and clicking "play" when we are ready to test the code. Contrast this to a typical workflow without
interactivity: code is added to a file using a text editor, and then executed in a separate window such as a terminal.
GUIs are in general more intuitive and less intimitading for beginners and those who are not primarily concerned with
producing code.

As we've already noted, Jupyter's GUI is centered around cells — these are the source of another major strength of
notebooks. Cells provide a user the ability to partition code into smaller chunks that can be run independently of one
another in any order, which is tremendously useful when writing and debugging a new data analysis. In our UFO sighting
example, we read in the input dataset in the first cell, clean it in the second cell, produce plots in the third and
fourth cell, and finally train and test a prediction algorithm in the fifth cell. The first step is time-consuming, but
in a notebook we only need to run it once, which saves the parsed data within the notebook. Thus we can run and rerun
subsequent cells as we build and test our analysis pipeline without having to go through the trouble of having to scrape
the dataset website each time.

Notebooks are also often used by researchers because they simplify sharing not only the final results of an experiment,
but also all the computational steps leading up to the results. Especially since notebooks can produce the full suite of
plots available in Python within the GUI, users can encapsulate an entire analysis, from data processing to
visualization, within a single file. This file can then be opened by anybody with access to a Jupyter server to
reproduce the experiment from start to finish. With an increasingly large pool of services offering access to such
servers — including Google Colab and Princeton's own myDella site — sharing notebooks with collaborators has never been
easier.


## When to consider using scripts? ##

This is a very important question. The easiest answer is that, often, there is no reason to. We have already considered
how convenient they make it to create and test an experiment from scratch, allowing you to separate different parts of
the experiment across "cells" for modular execution. Especially if you want to create plots quickly, notebooks' built-in
GUI support means that you can write code and produce plots within the same browser window.

Jupyter notebooks are great for experiments that are "linear" and "one-off", meaning that they consist of a single chain
of steps carried out one after the other, and that these steps will not have to be updated or rearranged at some point
in the future. Indeed, the very visual structure of a notebook reinforces this linear nature: one cell following
another, each executed in turn. One can of course choose one's own order of executing individual cells, but this will
usually result in errors, and notebooks do not have any built-in mechanism for informing which cell depends on another —
other than the aforementioned order of the cells themselves.

This linearity simplifies things, but it is also extremely limiting in terms of the kinds of experiments we can design.
Pipelines which execute heterogenous steps in parallel are off the table, as are pipelines which reuse code from other
pipelines without simply copying the text. The modular structure of notebooks is somewhat of an illusion; in reality,
the different cells have a very rigid relationship with one another, which is opaque to those who did not originally
create them.

Jupyter notebooks are also difficult to expand upon beyond the analysis they were designed to carry out. One of the more
obvious ways this problem manifests itself is when we try to parametrize an existing experiment. If we are e.g. training
a machine learning classifier with a regularization penalty of `alpha=0.01`, and we want to try other values of alpha in
a systematic way, there is no way of doing so without manually updating the stated value of `alpha` within the notebook.
For testing a handful of values of alpha this is fine, but notebooks quickly become cumbersome if we want to test
hundreds of such values. The penalty you pay for being able to execute individual notebook cells within a pretty GUI is
the inability to turn cells (or the entire notebook) into functions with arbitrary arguments and argument values.

Finally, notebooks' interactivity is a blessing and a curse.

In short, the time to consider moving beyond a notebook for your analysis is when you want to expand your experiment
beyond the exploratory stage. Notebooks become limiting when there are many possible hypotheses to test, or when you
need to run experiments that will take too long for manual supervision. Writing your experiment as a script or a package
will take more effort than a notebook, but an experiment of sufficient scope and scale will make such an investment
worthwhile.


## Final remarks ##

It is difficult to appreciate the full gravity of these considerations until one actually tries to port an experiment
from Jupyter. Thus an important feature of the subsquent chapters of this workshop is the exposition of the intermediate
steps involved in building a script, and then a package, from a notebook. This makes it easier to study the
considerations that must be made at each step of the process, and the trade-offs inherent in engineering one's code.
