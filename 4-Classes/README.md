## Introducing object-oriented design ##

![](../resources/calling.jpg)

The final part of this workshop will recast the code in our experiment around objects with attributes and actions, which
in Python (and most other languages supporting OOP) are called _classes_. Up to this point our we have expressed this
experiment as a series of isolated actions — this is particularly evident in the `main()` routine of our final script.
However, we can instead describe the work being done in terms of the behaviour of a `Sightings` object, which has
actions (_methods_ in the lingo of OOP) that use its internal attributes to produce the plots and classifier training we
need.
