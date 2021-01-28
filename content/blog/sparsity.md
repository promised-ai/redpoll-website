+++
title = "The Challenges of Missing Data and Sparsity
date = 2021-01-06
template = "post.html"
draft = true

[taxonomies]
categories = ["data science", "demo", "ai"]

[extra]
author = "Bryan Dannowitz"
subheading = "And how Redpoll's Reformer handles it in stride"
image = "gamma/magic-laser-calibration.jpg"
theme = "light-transparent"
+++

# Intro

* Talk about how rare it is to have an intact dataset
* Problematic values can be worse than no value at all
* Imputation techniques can help in a pinch
  * Mean imputation can be very ham-fisted, capturing no complexity. It's just a filler.
  * Sentinel values can be useful, but provide no info for a user beyond an indicator
  * Individual models can be made to predict missing values, but what happens when all features have this issue?

# Simple example

* Pick a new dataset
* Predict on a target
* Drop 10% of feature values
* Predict on target

# How much sparsity can Reformer handle?

* Perform the above test
  * Multiple plots
  * Y-axis: performance metric
  * X-axis: percentage of data missing
* No matter your sparsity, Reformer can handle it

# Usage: Movielens

* You couldn't typically use an off-the-shelf regressor on a sparse set like movielens
* Requires specialty methods like non-negative matrix factorization
* Reformer is up to task
* Pivot dataset from long-to-wide format
* Make some predictions on movie ratings on some holdout users

# Conclusion

* Bloh blah blah we will rock you

![fLength Residuals](/img/gamma/flength-residuals.png)
