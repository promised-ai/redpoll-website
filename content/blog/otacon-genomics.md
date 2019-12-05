+++
title = "Why we're using plant breeding to test the AI we're building for DARPA"
date = 2019-12-15
template = "post.html"
draft = false

[taxonomies]
categories = ["news", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "DARPA's SAIL-ON program focuses on novelty; plant breeding is a game of mastering novelty."
image = "red-flower.jpg"
theme = "light-transparent"
+++

"Defense AI" elicits images of satellite imagery overlaid with markers indicating the positions and movements of aircraft, troops, and supplies. Of War Games. So why is it that we're using plant breeding as a test bed to develop the AI we're building for DARPA?

In this post I'll discuss DARPA's SAIL-ON program, under which we're developing the OTACON platform, and discuss how plant breeding wrestles with all and more of the problems SAIL-ON hopes to address.

# The SAIL-ON Program

We're developing AI for DARPA under the SAIL-ON program. The stated objective of the SAIL-ON program is to make learning machines that are more robust to different types of *novelty* in the world. There are different types of novelty.

There are single odd examples. Say an image classifier trained to classify images of fruits receives an image of a toad. How will the machine react? A machine that knows only fruits, may pick up on the green bumpy skin and may, with high certainty, classify the toad as an avocado. What we would like the machine to do is to recognize the toad's weirdness and react to it. Perhaps by discarding the example, creating a new class of item for that example, or by asking a human for help. Recognizing novel examples is especially important for preventing attacks. An attacker may make changes to an image that are imperceptible to a human, but that change the classifier results dramatically (such as attacks that cause *stop* signs to be classifier as *speed limit 45* signs [FIXME: link]).

Novelty may arise from changes in the way the world is represented or in the dynamics of the world. When a person learns to play go, they typically learn on a small 9-by-9 board, and then eventually move up to a standard 19-by-19 board. A machine would have to learn 9-by-9 and 19-by-19 go as separate games; it could not transfer its knowledge from one board to the other. And what if you decide to keep score differently in the middle of the game, or a third player joins in? Again, these are things a human can handle with ease, but a machine must be retrained to do. 

## Sources of novelty

One of the things we need to do before we determine how well we can detect and react to novelty is to define what novelty is and where is comes from. SAIL-ON's *novelty hierarchy*, as laid out in the announcement, is a first attempt at this. I should note that many of the other performers have different thoughts about what novelty is and is not, and how it should be structured theoretically. Making these types of formalizations is outside my wheelhouse, so here I'll collapse the first draft hierarchy using my own hasty classification:

- **Class**: Previously unseen class or category of object (e.g. new dog breed or type of vehicle)
- **Features**: Change in how data features are specified or addition/removal of features (e.g. change of coordinate system).
- **Internal Dynamics**: Internal change in the system governing how the data are generated (e.g. gravity becomes stronger).
- **External Dynamics**: Changes in features that are not accounted for by the data that affect the collected features (e.g. environment)
- **Agent dynamics**: Changes in objectives or in the affect of an agent's actions

# Plant breeding

In plant breeding, we have a portfolio of plants from which we choose pairs to breed in order to achieve some objective &mdash; usually to improve performance. In corn, we might wish to maximize grain yield.

From our portfolio, we choose a set of pairs. For each pair, we grow both plants and breed them. From this breeding we achieve a set of *hybrids* which we plant, grow, and measure. The things we like, we advance, the things we don't like we forget about. Hybrids are usually subject to years of testing and further modification &mdash; via other breeding techniques like back-crossing, or biotech modification &mdash; before they become products and eventually find themselves as a part of the starting portfolio for future breeding.

## Molecular Breeding

Molecular breeding is basically using genetics data to inform breeding decisions. It is commonly used in plant and animals breeding. Genetics data are used for a couple of purposes: to get an idea of the performance of a cross without having to grow it, and to ensure genetic diversity. If we have enough data relating the genetics of plants to their phenotypes (observable traits) &mdhas; like grain yield and disease resistance &mdash; we can predict the phenotype from genotype. But selecting only the highest-performing individuals will likely reduce our genetic diversity. This is bad because if a new disease comes along that our crosses are not resistant to, they could be wiped out. We could lose everything. We need diverse genetic material for resistance.

Of course neither prediction nor diversity are as simple as that. Genetics are extremely complex and you'd need an unattainable amount of data to account for that complexity. And prediction is made more complicated by the environment which influences the development of organisms through epigenetic and external factors. 

Genetic diversity is a fairly well defined concept, but it is not the only type of diversity. In the age of machine learning, breeders must also breed for epistemic diversity &mdash; or diversity of knowledge created. That is, breeders must select crosses that sustain and improve learning.

## Novelty in the Breeding Process

Breeding is a game of novelty. Sometimes you seek it out and sometimes you fight with it. **The success of a molecular breeding program is determined by its ability to identify and characterize novelty**.

Performance criterion itself is based on novelty. A breeder is not looking for the cross with the highest predicted performance, but the highest probability of producing a high performer. Image that the performance of a hybrid follows a bell curve (see Figure FIXME). Let's say that hybrid A has high probability of performing well with little variance and that hybrid B is likely to perform poorer than A. Which do we choose? You may think A, but the answer is not so straight forward. What if B has a 10% to perform better 99.99% of the hybrids that A produces? Since we're talking plants we can take the scattergun approach; we can plant 100, and expect around 10 really awesome plants. This is why we seek out novelty, but the world of biology is filled with novelty that we must conquer to achieve our goals.


Evolution is fueled by novelty. When we make a cross we receive an entirely new set of genetics created by [recombination](https://en.wikipedia.org/wiki/Genetic_recombination) and mutation. When we develop the individual, the development of that individual is subject to environmental and epigenetic effects, which the sequence data do not account for. 

To relate back to the hierarchy:

- **Class**: A new gene would represent a new feature class and a new phenotype or set of phenotypes would represent a novel class.
- **Features**: Additional measurements or features. Often in breeding different measurements are taken depending on the performance of the individual. Some measurements are expensive or time consuming, and are not conducted on individuals that will be removed from the pipeline. Additionally, as sequencing technology changes, different genetic marker sets of different resolutions will be collected.
- **Internal Dynamics**: Movement of genes in the feature space. Several genetic mechanisms cause parts of DNA that influence certain phenotypes to move positions. This results in changes in the causal structure for parts of that data, which is something that is very hard for machines to deal with.
- **External Dynamics**: Environmental factors affect physical development through several mechanisms that present themselves differently depending on other environmental factors. A epigenetic effect might be easier to tease apart from plants grown in a growth chamber than plants grown in the field.
- **Agent dynamics**: Breeding objective change, for example in the face of a new disease or pest, and when trying to develop a new class of product, such as [short-stature corn](https://www.agriculture.com/news/crops/short-stature-corn-on-the-way-from-bayer-cropscience).

We're also generating our own training data, so the decisions we make not only influence our ability to deliver a great product and maintain diversity, but also influence our ability to make decisions in the future.

## Wrap up

Why we chose breeding as a platform to test the *general purpose* AI we're building for DARPA is clear: it's a game of characterizing and exploiting &mdash; mastering &mdash; novelty. And it's really really hard. It's so hard, and there are so many interacting factors, that it's like it doesn't want you to be able to do it...which is why we're excited to create an AI that can.

# Key Points
- DARPA's SAIL-ON program focuses on detecting and reacting to different types of novelty
- Plant breeding using genetic data is subject to all these types of novelty and then some
- The goals of plant breeding can be defined in terms of seeking out novelty
