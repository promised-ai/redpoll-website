+++
title = "Why you're not getting ROI from of data science"
date = 2019-10-20
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "business", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "It's not you, it's machine learning"
image = "black-rabbit-01.jpg"
theme = "light-transparent"
+++

**FIXME: Intro**

# Reasons ROI is hard in data science

The main reason that its hard to get ROI from data science is because machine learning is dumb and you are smart, but you think the opposite is true. Machine learning is requires a lot of clean, complete data, which you don't have. Machine learning puts the burden on you to know which data are important to answering your questions. Machine learning doesn't tell you what is real.

## Of pocket calculators and supercomputers

We've seen the great successes of AI over the last few years. It's beaten the best humans at chess, go, and Starcraft. And if it can do that, it should be able to tell me how to treat my sick patients, or which crops to plant in my fields.

The problem is that AI isn't playing the same game. In the case of chess and go, AI does search. It does a lot of [forward simulation](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search), thinking many, many moves ahead, and then choosing the action that is likely to lead to the best simulated outcome. People don't play that way. Brains are limited. We can only hold a handful of items in our active mind at a time. So we cannot brute force games like a computer can, the we must be clever because we are limited. We recognize formations of pieces and vulnerabilities in those formations. We look a couple of moves ahead down very specific paths and try to move out opponent into a position we know how to attack.

The same is true of Starcraft, but the dynamics are different. Starcraft is not a turn based game, it is played in real time. Now the computer has another unfair advantage: its ability to act and think thousands of times each second. This allows the machine to use units excel when given constant attention because it can essentially divide its attention. We cannot. We have to choose units that give more for less effort.

It is very strange to me that people are so quick to say that AI are better than people at games, but never say that calculators are better than people at math.

AI has been about building visually impressive demonstrations. We are made to believe that there are machines that exhibit higher intelligence and competency, and that doctors and lawyers will be automated away. The truth is AI is superficial and stupid, and we'd better resist being taken out of the loop for any tasks that matters.

## ML models can't cope with the state of your data

We're often out of luck before we even start doing data science.

Most machine learning models can't handle our data. Want to do deep learning? Better have millions of data or it can't learn. Better not have missing data because you're going to have to fill them in or throw them out. Better not have categorical data because you'll need even more data to support the extra encoding.  Do your data have unexplained trends in them? Were they collected from multiple sources, or before machine learning was a concern? It is probably going to learn to predict using that nonrandom noise.

## ML models can't do what you want with your data

**WRITEME**

## Asking questions takes too long

Most machine learning models are designed to answer one question: predict Y given X. What X and Y are depend on the model and what you're trying to accomplish. Y might be a numerical value like height, a categorical value like *good* or *bad*, or it might be the next few words in a sentence. The thing is that you have to know what X and Y are before you start even doing machine learning, which means you have to know which question you want to ask. The truth is that this you might not be able to ask the exact question you want because the model doesn't support that data. Maybe the kinds of data in X and Y are incompatible with the model that you believe will best answer your data. You have to tweak the data, which could bias the results; throw the incompatible data out, which is wasteful; or choose a new model that might not answer the question as well. Doing any of these things has changed the question.

The rest of the data science process is a cycle of training/fitting a model (the process by which the ML model is said to learn), validating, thinking you are not doing as well as you could be; then tweaking the model or choosing a new model entirely. And maybe even going back for different data.

By the time you have something that is as good as you think you're going to do, what you did looks nothing like what you wanted to do when you started because you've spend months manually trying to figure out which question is the one you can answer well. So you've answered one question well, reported the results, and now you're asked to answer another question. See you in a few months.

## Not knowing the which questions (if any) you can answer

A lot of the fitting, validating, tweaking cycle is due to people -- both decision makers and analysts -- refusing to believe they can't do something. The hype has you believing that machines are smart and you are dumb. That if you can't get actionable insights from your data, you are personally at fault. The truth is, sometimes you can't answer the question you want to. Sometimes you can't answer any questions at all. It might be because your data are bad. Or perhaps you do not have the right data to tease apart complex interactions. But it can also be that the thing you were hoping to find isn't a thing. You need to know that so you can stop wasting your time.

Data scientist and analysts often say something that fails to perform "doesn't work". Who is to say that poorly isn't the best you can do given the underlying (unknown) process that generated your data?

## Throwing science against the wall to see what sticks

What happens when something doesn't work, is that we find something that does. We tweak the data. We tweak the model. We find a new model; a more complex model. Maybe we stack and combine models. Maybe we extrude our data through a model so they can fit into another. Hell, if deep learning can beat the pros at starcraft, it can predict how long the tassels on our corn plant are going to be, right?

So we try everything. We spend a lot of time. And we force it to work.

## Not knowing what is real

And by the end of it we do not know whether what we found is real, or whether it is what we wanted to see. Even if we were not deliberately over-fitting our model, each thing we try is a coin flip. If we flip a coin enough times, it is going to come up heads eventually. In statistics, we correct for this. In machine learning, nobody asks.

## You are wasting your data scientists

Data scientists are expensive, so use them for things that can't be done by someone else more cheaply (someone who can do it faster and better at the same pay rate). If data scientists are collecting or cleaning data, or are doing discovery with packaged models, you are wasting your money.

Unless they are subject area experts, data scientists should not be in charge of choosing which questions to ask, which means they should not be doing data science discovery as outlined above. There are a number of companies offering tools that automate the kind of **bad** data science where one tries everything indiscriminately and keeps what is *best*. If you have subject area experts, use them to do discover. Scientists do discovery. For more on how to get more out of AI/ML in science check out this post. The gist of it is to train your experts in data science -- there is a lot more to know about, say genetics, than there is to know about data science.

**Data scientists should have one task: taking what was learned in discovery, sitting with experts, and building an optimized custom model to solve a specific business problem.** Every assumption in the model should be documented and signed off on by an expert. And the thing they build should be production ready. It shouldn't be thrown over the fence to engineering. But this isn't something any data scientist can do, it's something only a good data scientist can do.

## You have bad data scientists

A bad data scientist is a catalogue of models. They just go through all of the algorithms in whichever software package they like, and declare a winner based on raw performance.

A good data scientist has a number of qualities:

- They like coding and are good at it. They write production ready code that requires minimal integration effort.
- They understand statistics well enough to build their own models and understand the implications of their design choices.
- They will try things that make sense and are sound.
- They will not try things just because they are likely to work.
- They will tell you when something does not make sense or there is nothing to be learned.
- They will get mad at you if you try to make them do something unsound or to arbitrarily try some over-complex model you saw on a pop science website.

Good data science can come from a number of places. I think it's most common for great data scientist to be people that are experts in statistics and engineering, but some of the most impactful work comes from people who are excited about machine learning but also know their subject area front and back. Again, as I mentioned in my **FIXME: post** I very much believe that if you want the best bang for your buck, that you'll train domain experts in data science rather than train data scientists in a complex domain.

# Wrapping up

**WRITEME**

# Key Points
- A ML model can't handle your complex, sparse, partial, unstructured, and noisy data
- A ML model doesn't let you ask multiple questions quickly
- A ML model doesn't tell you what questions you can answer
- A ML model doesn't tell you what is real
