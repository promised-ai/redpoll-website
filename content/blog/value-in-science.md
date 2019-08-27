+++
title = "Three ways to get value out of AI/ML in scientific research"
date = 2019-07-31
template = "post.html"

[taxonomies]
categories = ["ai", "science"]

[extra]
author = "Baxter Eaves"
subheading = "Mainstream AI has neglected the needs of scientists, but that doesn't mean there is nothing for us to do."
image = "aquarium-aquatic-beautiful-2698871.jpg"
theme = "light-transparent"
+++

Modern AI has neglected science as a use case. Since the beginning of the latest AI summer, fueled by the emergence of Deep Learning, AI has focused on tasks like image processing, video processing, and games, which produce visually impressive demos, but are (for scientists) practically useless. What we have now are AI that require incredible amounts of data, don't generalize, and hide away their knowledge. This is bad for science. As a result science and research organizations don't much use deep learning. Not that they haven't tried. I've spoken to researchers and many organizations who have tried to use deep learning, but find it hard to get value out of it for a number of reasons: it need way more data than they have, it's hard to use, and it's uninterpretable (doesn't create knowledge).

More often than not, in research we cannot generate the millions of data needed to run a deep network. A clinical trial takes significant time and money; each data point could cost tens of thousands of dollars. A field trial for a new corn cultivar takes months and costs millions. We have to be efficient with our data.

Deep learning is hard to use. Post-2012 mainstream AI has been accused of being a [one-trick pony](https://www.technologyreview.com/s/608911/is-ai-riding-a-one-trick-pony/), consisting of infinite ways to construct and compose neural networks. To use them effectively -- to use most machine learning models effectively -- you must be an expert in it. You must be able to diagnose what is wrong when something goes wrong (ideally you'd be able to identify what might go wrong in the future), and you must understand how all the parameters of the network interact with the underlying optimization algorithm in order to properly fit a model. In science R&D organizations, this often means hiring expensive data scientists who are often more expensive than the subject matter experts, which means moving substantial resources away from product development into risky analytics projects. This is a non-starter in smaller organizations and those with narrow margins.

Most AI developed in the last decade are black box models. They hide their knowledge away. And even if you have a data scientist who can employ one of the many recently developed *explainability* techniques (see [this short review](https://link.springer.com/article/10.1007%2FBF00155763)), the knowledge inside is uninterpretable, [brittle](https://arxiv.org/abs/1710.08864), and, as a result, unactionable.

Then how do we use AI for scientific research? Well, if we can't use what there is, we have to rethink how we use AI an which techniques to use.

## Focus more on science and less on data science

In my eyes, the biggest problem with AI is that not everyone can use it. We have to hire someone to use it and hope they use it correctly. We bring in a data scientist to apply Deep Learning to some genomics data to find out that they didn't properly encode the data; that they misunderstood the value of "0" in our GBS data; or that they've been looking to answer a question that doesn't make sense in the domain. Data scientist require years of domain training in complex domains to be effective. You're not going to onboard an data scientist to molecular breeding, or drug screening, in a couple of weeks or even months. People require many years of training to learn to operate in these complex scientific fields. So let's use AI that these people can use. The scientists are the ones who know that most about the nuances in the data, who know the most about the impactful problems, and who -- most importantly -- know the right questions to ask to solve those problems and to learn.

In order to do this, use interpretable models. If there is a well-studied, and accepted process describing the mechanism by which your data were generated, build a model around that, or (better) use existing code if it exists. If there is ambiguity around the mechanism, employ simple models like [general linear models](https://en.wikipedia.org/wiki/General_linear_model), or [graphical models](https://en.wikipedia.org/wiki/Graphical_model) constructed by structure learning algorithms. Structure learning algorithms use techniques to identify dependencies in your data. The user can then define a model for those dependencies using a probabilistic programming language like [Stan](https://mc-stan.org/) or [PyMC3](https://docs.pymc.io/). Then you have you own mechanistic model.

Of course, you need people who are able to do that. If you have scientists who can code, you already have them. They'll just need training in these methods. Most scientists already have at least a casual understanding of basic statistical methods. It's not a huge stretch from simple linear regression to multiple regression. And Bayesian statistics is about the easiest mathematical concept there is, and there are a number of great tutorials aimed at folks with only a passing understanding of classic, p-value-based statistics (tutorial [pdf](https://www.cell.com/cms/10.1016/j.tics.2006.05.006/attachment/07501a2d-51bf-45cd-a524-170960d1dccc/mmc1.pdf)). The hardest part will be getting folks up and running on a probabilistic programming language. If they're coding already, it's just a matter of going through the tutorials.

It will be worth it. In my experience it's faster to get a domain scientist to a functional level of data science than it is to get a data scientist to a functional level in a complex scientific domain.

## Use interpretable AI models

If our organization sells a science based product, we require knowledge to innovate. Without advanced AI or machine learning, we generate knowledge the old fashioned way: by designing and conducting experiments. Since a human is in charge of the whole process, the process generates human interpretable knowledge that can be communicated in human language. This means we can share it with other scientist, stakeholders, and budget makers. This means we can have conversations about it with our friends and colleague, who can question it and we can develop new ways to look at the data, and new experiments to run. This ensures continual improvement and innovation. You know, science. When we use a black box model to make a prediction, we get an answer that we are forced to either accept or reject. There is no reasoning or process to scrutinize or draw inspiration from; no conversations to be had. The scientific process is stifled; program and budgeting decisions feel icky because they are uniformed. 

If there is no way for a human to extract and communicate, the untouched and untranslated knowledge from a machine, that machine is not suitable for use in science.  The best case scenario is that there is nothing to talk about because nothing was retrieved. The worst case scenario -- enabled by many of the explainability techniques developed in the past few years -- the knowledge is misrepresented and over trusted, causing efforts to be allocated to fruitless or dangerous endeavors.

## Use AI to design experiments

When we do science we generate our own data. These data are then fed back into our AI or ML models, which may then suggest which data to generate next. For example we clone some cells thousands of times hoping to generate a disease resistance mutation. We expose the cloned cells to a disease then collect the genetic sequence and examine the health of those cells that survived. We feed all those data to a model that tells us which cell is best to clone next. We repeat the process. After a while the model becomes entrenched. It has learned only about the good things, and has not explored. Its knowledge, and thus our knowledge, is full of gaps. To fill those gaps, we must generate data on potentially bad or risky cells. How can we make predictions about bad or risky cells if we haven't learn about bad or risky cells? But of course the folks writing the checks protest. How do you measure the benefit of collecting data about things that are risky or evidently bad? You stop thinking about data, and start thinking about information.

The unit of information is the 'bit'. If you have an interpretable model of your data you can simulate an experiment. You then define an objective for the experiment. Most of the time you'll define the success of the experiment in terms of information gain, which is the number of bits of information you learned from the experiment. You can add additional parameters to your optimization objective. If experiments have variable monetary cost, then you can define your experiment objective in terms of bits of information divided by cost. Now you can choose experiments to optimize your learning per dollar spent.

## Wrap up

Just because modern AI is largely unacceptable for science, that doesn't mean that there's no way to make value from AI in science. We can avoid taking resources away from product development by training the scientist we already have in interpretable AI techniques. We can limit the cost of and time burden of training by focusing on a small set of flexible and effective models. And we can get more out of our data and experiments by using these models to predict the information gain and cost of our experiments.


## Key points

- Most modern AI are data hungry, uninterpretable, hide their knowledge, and are difficult to use making them ineffective in science.
- Science organizations will get more out of analytics by training their domain experts in model building since they're the people who know the most about the problem.
- To reduce the effort in training, focus on classes of probabilistic models that scientists have some exposure to.
- Use models that can be interpreted as they are without need for addition translation. This allows easier communication between scientist and stakeholders, and permits broad input and diverse ideas.
- AI learn from the data they're given so they need to right data to learn from.
- An interpretable AI model can simulate experiment which allows users to determine which experiments optimize the learning/cost ratio.

{{ signup(heading="Sign up for updates") }}
