+++
title = "Why ROI from data science is hard"
date = 2019-11-29
template = "post.html"
draft = true

[taxonomies]
categories = ["data science", "business", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "Machine learning isn't there yet, but neither are we."
image = "black-rabbit-01-de.jpg"
theme = "light-transparent"
+++

Data science is an expensive endeavor — the personnel are expensive, the infrastructure is expensive — so getting return on that investment is hard to begin with. The problem is worse in safety intensive and knowledge hungry domains; machine learning and AI were mostly not designed to handle the task constraints of these domains, so you cannot usually toss data into off-the-shelf software and expect a valuable/impactful solution. In these tasks, more manual data science with more input from more specialist is required over a longer period of time.

In this post, we'll discuss the problems with machine learning that make it difficult to discover valuable and reliable knowledge, and some ways that data scientist are often misused that increase costs or diminish returns.

Over the course of this post, I will discuss a data science anti-pattern — try everything until it works — which certainly isn't universal, but which is common, and which we've all fallen into one time or another. The discussion of this anti-pattern is not meant to cast blame, it is meant as a reminder that we need to be self-aware and avoid falling into this trap — or to push back when asked to do things singly for the sake of boosting a metric.

# Machine learning isn't where it needs to be

For all the hype, machine learning just isn't where it needs to be, especially when we're talking about human-in-the loop domains where knowledge and safety are the keystone of successful operations. Machine learning does little to help you through discovery. On its own, it doesn't help you to figure out which questions you can answer, it doesn't tell you where your knowledge gaps are or how to fill them, and it doesn't do much to tell you whether what it learned is real or reliable.

And there are odd psychological phenomena stemming from the innumerable performance metrics and the dopamine boost we get from optimizing them.

Combined, the complexity and psychology surrounding ML often traps us in cycles where we try something, it fails, we tweak it or try something else entirely until something "works". Cycles that takes months or longer to escape, and that in the end often produce systems that we cannot be sure will not fail spectacularly in production.

Part of the problem is machine learning. Part of the problem is us.

## ML models can't cope with the state of your data

We're often out of luck before we even start doing data science.

Most machine learning models cannot handle our data. Want to do deep learning? Better have millions of data or it can't learn. Better not have missing data because you're going to have to fill them in (predict so you can predict) or throw them out. Better not have categorical data because you'll need even more data to support the extra encoding. Do your data have unexplained trends in them? Were they collected from multiple sources, or before machine learning was a concern? The model is probably going to learn to predict using that nonrandom noise.

## Asking questions takes too long

Most machine learning models are designed to answer one question: predict Y given X. What X and Y are depend on the model and what you're trying to accomplish. Y might be a numerical value like height, a categorical value like *good* or *bad*, or it might be the next few words in a sentence. The thing is that you have to know what X and Y are before you start even doing machine learning, which means you have to know exactly which question you want to ask. The truth is that this you might not be able to ask the exact question you want because the model doesn't support that data. Maybe the kinds of data in X and Y are incompatible with the model that you believe will best answer your data. You have to tweak the data, which could bias the results; throw the incompatible data out, which is wasteful; or choose a new model that might not answer the question as well. Doing any of these things has changed the question.

The rest of the data science process is a cycle of training/fitting a model (the process by which the ML model is said to learn), validating, thinking you are not doing as well as you could be; then tweaking the model or choosing a new model entirely. And maybe even going back for different data.

By the time you have something that is as good as you think you're going to do, what you did looks nothing like what you wanted to do when you started because you've spent months manually trying to figure out which question is the one you can answer well. So you've answered one question well, reported the results, and now you're asked to answer another question. See you in a few months.

## Not knowing which questions (if any) you can answer

A lot of the fitting, validating, tweaking cycle is due to people — both decision makers and analysts — refusing to believe they can't do something. The hype has you believing that machines are smart and you are dumb. That if you can't get actionable insights from your data, your inability is to blame. The truth is that sometimes you cannot answer the question you want to. Sometimes you cannot answer any questions at all. It might be because your data are bad. Or perhaps you do not have the right data to tease apart complex interactions. But it can also be that the thing you were hoping to find isn't a thing at all. You need to know that so you can stop wasting your time.

Data scientist and analysts often say something that fails to perform "doesn't work". Who is to say that poorly isn't the best you can do given the underlying (unknown) process that generated your data?

## Throwing science against the wall to see what sticks

What happens when something doesn't work, is that we find something that does. We tweak the data. We tweak the model. We find a new model; a more complex model. Maybe we stack and combine models. Maybe we extrude our data through a model so they can fit into another. Hell, if AI can beat the pros at starcraft, it can predict how long the tassels on our corn plant are going to be, right?

So we try everything. We spend a lot of time. And we force it to work.

## Not knowing what is real

If we go down the path of just trying everything, by the end of it we often cannot know whether what we found is real, or whether it is what we wanted to see. Even if we were not deliberately over-fitting our model, each thing we try is a coin flip. If we flip a coin enough times, it is going to come up heads eventually. If we try and tweak enough models, we'll get one that does what we want. In statistics, we correct for this. In machine learning, nobody asks.

# Misuse, or unnecessary hiring, of data scientist

This point deserves a post of its own, but the long and short of it is that data scientists are expensive. If you're small, their salaries, fringe, and overhead might be eating any return they generate. Don't use them if you don't need to, and use them well if you do.

## Do not use data scientists to do things that can be automated or done more cheaply by someone else

There are a number of companies that offer products that automated the awful practice of try-everything-until-something-works analytics, so you don't need to hire someone to do it. You can simply hire someone to clean the data enough for it to be dropped into one of these solutions.

And there are a lot of (mostly) non-automateable tasks that data scientists do that they ought not be doing. For example, you might have engineers that can pull and process data much faster; or you might have domain scientists that would be much more effective at exploration and discovery.

## Hire data scientists who are skilled developers and who understand statistics

A good data scientist does not toss garbage code over the fence for engineering to productionize (doubling the time to production), but is someone who writes production quality code.

A good data scientist is not a catalogue of machine learning models, but is someone who can build a tailored model to effectively solve a specific business problem. Your competitors are already using premade machine learning models. If you want to do better you must innovate. If your data scientists understand statistics and model building, they can do this for you.

## Hire data scientist that communicate well

A data scientist is a translator. They direct a machine to learn something, but the machine cannot communicate, and the way it stores its knowledge is not always easily interpreted by a human being. The data scientist converts that strange machine knowledge into plain, actionable language that can be understood by decision makers. The more complex the problem space, the more difficult this will be. This is one of the reasons why I advocate for using domain experts for data science in complex domains.

# Wrapping up

**WRITEME**

# Key Points

**FIXME: Needs love**

- Most machine learning is too rigid and opaque
    - It can't handle your complex, sparse, partial, unstructured, and noisy data
    - It takes too long to ask a lot of different questions
    - It doesn't tell you what questions you can/cannot answer
    - It doesn't tell you what is real
- Do not hire data scientists unless you really need them and you are certain the value of the problems you are wanting to solve far outweighs their expense
- Do not use data scientists to do things that can be automated (trying all the old standard ML models) or done more cheaply by someone else (like data collection and cleaning)
- Hiring data scientist that code well ensures high quality and reduces time to production
- Hiring data scientists with domain experience improves communication, and thus improves decision makers' ability to act confidently.
