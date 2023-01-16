## Converting a Jupyter notebook into a script ##

![](../resources/x-files-analysis.jpeg)

We've decided that our UFO sightings experiment has enough merit to warrant deeper analysis. This means that it's time
to consider the tools available in Python to make expanding our analysis code easier, starting with command-line
scripts.

The scripts included in this chapter should be run from within an activated `notebooks-packages` conda environment; see
documentation within each script and the README at the top of this repository for further details.


### A) Converting a notebook into a script ###

Making a script is not that hard; all we've done here is concatenate the cells of our notebook into one block of
code, and add some small stylistic updates. We put this block in a `main()` function, and then we told the script to run
this function when it is being invoked from the command-line (`if __name__ == '__main__':`). That is, when we use the
command `python A_predicting-ufo-sightings.py`, Python runs the code in the script with a global variable `__name__` set
to the value `"__main__"`.

Did we gain anything from this? Not really! It's now easier to run this experiment from start to finish with one push of
a button, but it's not like Jupyter notebooks make that particularly difficult either. We are no longer constrained to
running our analysis on a Jupyter server, and we can run this script in any terminal with the right conda installation,
but that is about it. On the other hand, we have lost the ability to quickly look at our plots as our analysis runs, and
it's not immediately obvious how we would step through this code to examine intermediate outputs or to debug[^1].

Excepting some specific circumstances, creating a script like this is pointless. We did not put much time into it, but
it has not made our lives easier in any significant way, nor has it made our experiment any more compelling. This is
because we have neglected to actually use any of the Python features that make scripts powerful and versatile! Let's
take a look at two possible ways we can build upon both our code and our experiment.


### B) Using argparse to parametrize our script ###

The command `python A_predicting-ufo-sightings.py` is conspicuous in its brevity. For all the different modications and
tweaks we could make to our analysis, all it does run one particular version of the sightings pipeline, one particular
way. Running from the command line means we could append any arbitrary text to the command, impossible within a notebook
— can we leverage this fact?

Fortunately Python has a [package](https://docs.python.org/3.9/library/argparse.html) `argparse` that allows a script to
accept arguments from command line — that is, to parse text appended to the command
`python A_predicting-ufo-sightings.py` into variables that can be accessed by code within the script[^2]. In this script
we are only using a small slice of the many possible ways `argparse` can parse input arguments, and yet we have already
made out analysis considerably more flexible and interesting.

`argparse` implements a two-tiered structure for passing arguments to a script, in which _positional_ arguments are
followed by _optional_ arguments. Positional arguments, such as `years` in our script, must **always** be specified, and
specified in a particular order, but they do not need to be preceded by the name of the argument. This should be used
for the small number of attributes (ideally up to three or four) in your experiment that are at the root of its
behaviour and that do not have obvious default values.

Optional arguments must be preceded by the name of the argument, but this also allows them to be specified in any order,
as long as they are placed after whatever positional arguments are required. This more explicit structure also allows
for far greater flexibility within each argument: we can define optional arguments that are a list of arbitrary length
(`--states`), a boolean flag that can be "turned on" (`--create-plots`), a flag that can be specified an arbitrary
number of times to "increase" its strength (`-vvvvv`), and of course many other options that we omit here.

By creating an explicit list of attributes in our experiment that can be manipulated we have also given the user some
very useful information about the structure of the experiment itself. Writing help messages for each argument further
helps document what this experiment is testing. `argparse` automatically compiles this information into a help screen,
which can be accessed by running `python B_predicting-ufo-sightings.py -h`.


### C) Using functions to modularize our script ###

`argparse` makes life easier for the user of your script, who can now find many different ways to modify and test our
sightings prediction pipeline. What about making life easier for you, the designer and maintainer of this code?[^3] Our
experiment is still one long block of code, but breaking it up into smaller parts that interact with one another will
simplify debugging and adding functionality to this analysis.

We were careful in designing the cells of our notebook, and so the five original cells correspond directly to the steps
of processing the input datasets, cleaning them, creating two types of plots, and training a prediction algorithm. This
proved convenient, as each cell became a self-contained portion of our analysis. In our first two scripts we made
demarcations in the code using block comments to identify where these steps were taking place; but it would be better if
the structure of the code itself reflected the structure of the experiment in a more meaningful manner.

In this script we thus created separate functions for each of our steps (`scrape_sightings()`, `plot_totals_map()`,
etc.) Each of these is invoked by `main()`, which runs as before after parsing arguments from command line. This new
modular design makes it easier to see what steps are being carried out by our experiment, and in what order, without
having to pore through the minutae of what each individual step does.

The new functions are also useful for giving us greater flexibility around extending the structure of the experiment.
For example, to repeat a step, now we just have to call the function again, instead of copying the code, or running the
script again. We take advantange of this — in conjunction with another useful feature in `argparse` — to add the ability
to test multiple sets of states in one run of the script by allowing the `--states` argument to be repeated.

Another advantage of modularization becomes more apparent when other researchers would like to use parts of your
experiment without having to run the whole thing. Debugging code for scraping websites can be a pain, and so your
colleague may want to try their own experiment on the same sightings data without having to write their own
`scrape_sightings`. This design makes it easier for them to identify the part of the code responsible for this step, and
to copy it in its entirety without having to worry about what is happening in the rest of the script.


[^1]: debugging tools abound but are outside the scope of this workshop: the built-in Python debugger is sufficient in
      most circumstances: [pdb](https://docs.python.org/3.9/library/pdb.html)
[^2]: there are more advanced alternatives to `argparse`, which you can read about
      [here](https://realpython.com/comparing-python-command-line-parsing-libraries-argparse-docopt-click/);
      to read in the arguments as a flat list you can also use the more basic `sys.argv` approach
[^3]: in the context of research, 99% of the time the sole user and also designer of the code will be: you
