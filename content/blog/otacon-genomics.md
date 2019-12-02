+++
title = "Why we're using molecular breeding to test the AI we're building for DARPA"
date = 2019-12-15
template = "post.html"
draft = false

[taxonomies]
categories = ["news", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "DARPA's SAIL-ON prgram focuses on novelty and novelty is the mechanism of evolution"
image = "red-flower.jpg"
theme = "light-transparent"
+++

"Defense AI" elicits images of satellite imagery overlaid with markers indicating the positions and movements of aircraft, troops, and supplies. Of the real-time game of war. So why is it that we're using plant breeding as a test bed to develop the AI we're building for DARPA?

In this post I'll discuss the goals of DARPA's SAIL-ON program, under which we're developing the OTACON platform, and discuss how plant breeding wrestles with all and more of problems SAIL-ON hopes to address.

# The SAIL-ON Program

We're developing AI for DARPA under the SAIL-ON program. The stated objective of the SAIL-ON program is to make learning machines that are more robust to different types of *novelty* in the world. There are different types of novelty.

There are single odd examples. Say an image classifier trained to classify images of fruits receives an image of a toad. How will the machine react? A machine that knows only fruits, may pick up on the green bumpy skin and may, with high certainty, classify the toad as an avocado. What we would like the machine to do is to recognize the toad's weirdness and react to it. Perhaps by discarding the example, creating a new class of item for that example, or by asking a human for help. Recognizing novel examples is especially important for preventing attacks where an attacker can make changes to an image that are imperceptible to a human, but that change the classifier results dramatically (such as attacks that cause *stop* signs to be classifier as *speed limit 45* signs [FIXME: link]).

There is novelty stemming from changes in the way the world is represented or in the dynamics of the world. When a person learns to play go, they typically learn on a small 9-by-9 board, and then eventually move up to a standard 19-by-19 board. A machine would have to learn 9-by-9 and 19-by-19 go a separate games; it could not transfer its knowledge from one board to the other. And what if you decide to keep score differently in the middle of the game? Again, this is something a human can handle with ease, but a machine must be retrained to do. 

## Sources of novelty

The first draft of SAIL-ON's *novelty heirarchy* outlines several sources of novelty that I have collapsed:

- **Class**: Previouslly unseen class of object
- **Features**: Change in how data features are specified or addition/removal of features
- **Internal Dynamics**: Internal change in the system governing how the data are generated.
- **External Dynamics**: Changes in features that are not accounted for by the data that affect the collected features (e.g. environment)
- **Agent dynamics**: Changes in objectives of in the affect of an agent's actions

# Breeding

The goal of breeding is to choose pairs of parents to cross (breed) that produce the set of crosses that produce the best outcomes. In corn, we might wish to maximize grain yield.

## Molecular Breeding

Molecular breeding is basically using genetics data to inform breeding decisions. It is commonly used in plant and animals breeding. Genetics data are used for a couple of purposes: to predict the performance of an individual without having to grow it, and to ensure genetic diversity. If we have enough data relating the genetics of plants to their phenotypes (observable traits) -- like grain yield and disease resistance -- we can predict the phenotype given a genotype. But selecting only the highest-performing individuals will likely reduce our genetic diversity. This is bad because if a new disease comes along that our individuals are not resistant to, we could lose everything. We need novel diverse material for resistance.

Of course neither prediction nor diversity are as simple as that. Genetics are extremely complex and you'd need an unattainable amount of data to account for that complexity. And prediction is made more complicated by the environment which influences the development of organisms through epigenetic and external factors. 

Genetic diversity is a fairly well defined concept, but it is not the only type of diversity. In the age of machine learning, breeders must also breed for epistemic diversity -- or diversity of knowledge crated. That is, breeders must select crosses that sustain and improve learning.

## The Breeding Process

The process, in simple terms, goes like this:

1. From a portfolio of individuals, select a set of candidate crosses that maximize the objective
2. Make and grow the crosses
3. Measure and score the crosses
4. Return the set of best scoring crosses to the portfolio
5. Go to 1.

There are a lot of sources of novelty here. Biology is fueled by novelty. When we make a cross we receive an entirely new set of genetics created by [recombination](https://en.wikipedia.org/wiki/Genetic_recombination) and mutation. When we develop the individual, the development of that individual is subject to environmental and epigenetic effects, which the sequence data do not account for. 

## Novelty in breeding

To relate back to the hierarchy:

- **Class**: A new gene would represent a new feature class and a new phenotype or set of phenotypes would represent a novel class.
- **Features**: Additional measurements or features. Often in breeding different measurements are taken depending on the performance of the individual. Some measurements are expensive or time consuming, and are not conducted on individuals that will be removed from the pipeline. Additionally, as sequencing technology changes, different genetic marker sets of different resolutions will be collected.
- **Internal Dynamics**: Movement of genes in the feature space. Several genetic mechanisms cause parts of DNA that influence certain phenotypes to move positions. This results in changes in the causal structure for parts of that data, which is something that is very hard for machines to deal with.
- **External Dynamics**: Environmental factors affect physical development through several mechanisms that present themselves differently depending on other environmental factors. A epigenetic effect might be easier to tease apart from plants grown in a growth chamber than plants grown in the field.
- **Agent dynamics**: Breeding objective change, for example in the face of a new disease or pest, and when trying to develop a new class of product, such as [short-stature corn](https://www.agriculture.com/news/crops/short-stature-corn-on-the-way-from-bayer-cropscience).

We're also generating our own training data, so the decisions we make not only influence our ability to deliver a great product and maintain diversity, but also influence our ability to make decisions in the future.

It becomes clear why we chose breeding as a platform to test the *general purpose* AI we're building for DARPA: it's really really hard. It's so hard, and there are so many interacting factors, that it is like it doesn't want you to be able to do it...which is why we're excited to create an AI that can.

# Key Points
- DARPA's SAIL-ON program focuses on detecting and reacting to different types of novelty
- Plant breeding using genetic data is subject to all these types of novelty and then some
