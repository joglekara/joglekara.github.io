---
layout: post
title: "Sparse Dense Multiplication in Tensorflow"
author: "Archis Joglekar"
categories: journal
tags: [Tensorflow, Sparse Matrix, Scientific Computing, OSS]
---

Sparse-matrix-based computations are an important subset of scientific computing. For example, performing
finite-difference using matrix math quickly becomes unfeasible if performed using dense operations.

As of v2.1, Tensorflow only supported matrix multiplies $AB$ such that only $A$ is allowed to be sparse. At Noble.AI,
we wished to have the ability to perform sparse-dense multiplications in any order. To do so, rather than
maintaining an in-house fork of Tensorflow, we decided to contribute to the repository.

I learned quite a few things here. Tensorflow's contribution guidelines require creating email groups, preventing API
changes, and many other details that took a few iterations to absorb completely.

However, as of v2.2, this Pull Request has been merged. We also published a notebook motivating, detailing, and testing
our implementation [here.](https://github.com/noble-ai/open-source-contributions-unit_tests-tutorials/blob/master/notebooks/02-11-2020%20-%20Multiplying%20a%20Dense%20array%20by%20a%20Sparse%20array%20in%20TensorFlow.ipynb)