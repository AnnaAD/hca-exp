## Hierarchical Clustering

I am running a few cpp experiments regarding incorperating heuristics and parallelization into hcn.

## hclust-cpp

Included in this directory is the following repository: [https://github.com/cdalitz/hclust-cpp/](https://github.com/cdalitz/hclust-cpp/)

This repository is an adaptation of Daniel Mullner's work:

> "fastcluster: Fast Hierarchical, Agglomerative Clustering Routines for R and Python." Journal of Statistical Software 53, no. 9, pp. 1-18 (2013)

This version is a standalone c++ library by Christoph Dalitz, [cdalitz](https://github.com/cdalitz).

Changes made:
- include/build with openmp.
- add timer via include dir.

## scc

This is an implementation of scc in python.

Changes made: 
- `.A` syntax for matrices no longer supported in numpy for certain types of matrices??
    - switched to `.toarray()` when complaints arrose

`pip install .`
installs scc based on setup.py file.


## graphgrove

Changes made:
- how to include Eigen library OR the C

`pip install .`
- builds graphgrove cpp
- installs python project


