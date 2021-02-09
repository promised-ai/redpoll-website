+++
title = "Escaping the data science death spiral"
date = 2021-02-09
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "How today's Machine Learning technology forces us into wasteful processes and how we can escape them"
image = "skull-01.jpg"
theme = "light-transparent"
front_page = false
+++

After three years in development, the recommendations of a multi-million-dollar decision support project are delivered to a committee of decision-makers who patently ignore them.
A nine-figure data-science company lays off 60% of its workforce after two years failing to develop a Machine Learning (ML) solution that can bring value to its target market.
A hospital scrubs its efforts to integrate clinical support software after failing to make useful recommendations, more than three years and $60 million later.
Mega-scale data science waste is commonplace.

The data science process is hard.
It is inefficient, error-prone, and fragile.
It is full of iteration and cycles: we do not like what we are producing, so we take a few steps back, tweak, and try again.
Every step requires communication with subject area experts and stakeholders; it requires those subject area experts to stop their work to ensure that our work is valid.
It requires careful archival and documentation of all previous action to ensure reproducibility, so we can continually progress toward the goal and avoid tripping over our past failures.
Cycles are fragile. Every cycle presents an opportunity for the process to fail.
This fragility incurs a great deal of waste &mdash; from the work required to support ML infrastructure, to the domain-specific ML solutions.

# The standard data science process: CRISP-DM

If you are a data scientist, you probably use an analytics process called [*Cross-industry standard process for data mining*](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) or CRISP-DM.
You might not be aware what you do is called "CRISP-DM", but this is how everybody does analytics because it is the only workflow that makes sense with today's Machine Learning technology. It goes like this:

<style>
    .row {
        padding: 1rem 0;
        align-items: center;
    }

    .row div:nth-child(2) {
        font-size: 1.3rem;
    }

    .col-left {
        background-color: #f2f2f2;
    }

    .col-right {
        padding: 1rem;
    }
</style>

<br>

<div class=row >
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-01.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>You start by identifying a question that would be valuable to answer.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-02.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then you must find data that support answering your question.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-03.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>However, it turns out you do not have &mdash; or cannot get &mdash; the data you need, so you must reformulate the question to reflect the state of the data.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-04.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then you choose from one of infinite Machine Learning models, each with a legion of parameters, hyperparameters, and optional add-ons.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-05.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Nevertheless, the model you chose does not support your data, so you have to transform the input. You may need to:
            <ul>
                <li> Encode your categorical values as continuous (e.g., one-hot)</li>
                <li> Transform data that lie across unfavorable support</li>
                <li> Transform data that do not adhere to the model's assumptions, such as normality</li>
                <li> Fill in missing data or throw out entire records with missing values</li>
            </ul>
        </p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-06.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Maybe you cannot find a model that answers the exact question you want to ask with the data you have.</br></br>You'll have to ask an adjacent question.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-07.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Once you get the question, your data, and the first-try model sorted out, you can train the model.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-08.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Now, you validate/score the model.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-09.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>The model does not perform. Things rarely work on the first try.</br></br>So you tweak the model or select a different model entirely.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-10.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then one of two things happens: after several iterations you hit a performance optimum, after which further effort is unrewarded; or you hit your deadline. In either case, it is time to report.</br></br><strong>Congrats! &#129395; You have answered a question!</strong></p>
    </div>
</div>
<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-11.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Your stakeholders have new questions. Time to do it again.</p>
    </div>
</div>

Things to note:
- There are nearly as many paths backward as there are paths forward.
- It takes a significant amount of iterative work to answer a single question.
- People are doing most of this work; not AI.
- Sometimes you cannot answer the question because the information does not exist in the data.
- The question you end up answering is often not the one you set out to answer.

The process eventually begins to resemble a game of *Chutes and Ladders*.
If you are lucky, you will always progress &mdash; up the ladders &mdash; toward an answer.
However, we inevitably step into one of the many modeling pitfalls &mdash; falling down chutes &mdash; having to iterate and try again. Unfortunately pitfalls are built into the process, so we can expect to fall a lot.


## The Death Spiral

The whole data science cycle is rough, but the model - train - validate process is a Death Spiral.
This cycle is where I have seen the most data science projects go off the rails.
Many elements collide to make this part of the process particularly treacherous, but most of the trouble is in measuring and maximizing performance.

### You can't prove a negative

Machine learning does not give you a good way to know whether your question is answerable with your data or how well you should expect to do.
The first time you try a model, you are likely to fail.
Your options are then to find a model that will work, or to prove that no model will work.
You must succeed, or you must prove unicorns do not exist.
Under some circumstances, if you have the math and the problem is well-behaved, you may be able to mathematically prove that modeling is impossible.
But, alas, in my experience, mathematical proof is not compelling to stakeholders.
People want and understand empirical results.
In practice, the only way to prove that no model will work &mdash; or no model will work better &mdash; is by deduction: to try *everything*.

Thanks to the speed of research (there were 163 AI papers published to arXiv in the first 4 days of February) and combinatorial expansion, there are innumerable things to try.
And since none of these things quite fit your question and your data, there is always something to blame.
"Well, this model does not work well with categorical data, so maybe we could try a different encoding or embedding", "This deep network only has two layers, so it's probably not very expressive", or "We have a lot of missing data, and we've only tried this one imputation method".

### Communicating can cost as much as not communicating

In complex domains, like health, biotech, and engineering, where there is extensive science involved and high cost for failure, domain experts *must* be involved in every step of the process (apart from training).
Each step requires e-mailing an expert, scheduling time &mdash; days or weeks out &mdash; for them to stop what they are doing, and then sitting with them so they can tell us what is wrong and what is right.
But communication is a bottleneck. There are two options

1. We communicate properly, as frequently as CRISP-DM requires, and risk both annoying our experts and dragging out our project out by months (or longer).
2. We do not communicate and risk delivering something likely to fail or to be rejected by decision-makers.

Stakeholder rejection may rightly occur because, in failing to consult decision-makers, we have not understood their needs and have delivered something unwanted or unusable; or because we have made something opaque (Machine Learning is opaque by nature), which they do not understand and cannot be expected to trust.

As a result of all of this, it is common for data science projects to be shelved after months or even _years_ of investment and development.

### Why does the data science Death Spiral exist? 

The Data Science Death Spiral exists because Machine Learning is, and always has been, focused on modeling questions.
What is the value of Y given X?
Which data are similar?
What factors determine Z?
Machine learning assumes the user has a well-defined problem with nice, neat, and complete data.
This is rarely the case.

Because Machine Learning focuses on modeling individual questions, the standard data science process must focus on modeling questions; so the standard process is good when you have a well-defined problem and nice data.
It is bad when you have a nebulous problem and ugly data, and is disastrous when exploring. And if it is disastrous for exploration, it disastrous for _innovation_.

# A better data science process through humanistic systems

People do not model questions.
You do not have to know what you want to learn before you learn it.
You go out into the world, you observe data, and learn from those data.
You learn about the process that produced the data: the _world_.

What if we had Machine Learning or artificial intelligence technology focused on modeling the **whole data** rather than just modeling single questions?
We would be able to answer any number of questions within the realm of our data without re-modeling, re-training or re-validating.
We would know which questions we'd be able to answer and how well.

The data science process would look like this:

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-01.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Get a vaguely coherent dataset.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-02.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Learn a causal model that explains the data.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-03.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Ask a question.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-04.png" style="max-width: 200px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Ask more questions.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-05.png" style="max-width: 200px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Report to stakeholders.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-06.png" style="max-width: 240px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Ask their questions.</p>
    </div>
</div>

<!-- <div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/human-07.png" style="max-width: 240px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Optionally: update, edit, fill in, or extend your data and ask questions of the updated data. This is a capability unique to <a href="/blog/stream-native/">stream-native systems</a>.</p>
    </div>
</div>
-->

The first thing to notice is that there are far fewer steps and that every step takes us closer to where we want to be.
The key in this process is asking sensible questions.
Asking sensible questions requires deep domain expertise, so it would be most effective for this Data Science process to be run by a domain expert.

Eliminating iteration through up-front learning and turning a *question-modeling* system into a *question-answering* system significantly streamlines discovery and reduces time to production.
Under this process, discovery-to-production takes hours or days rather than months or years.

**And the best part is: this technology exists today. You can check out what this looks like in practice <span style="color: crimson">[here](@/redpoll-vs-ml.md)</span> and <span style="color: crimson">[here](@/blog/gamma.md)</span>.**

When we consult for clients, we take their data, feed it through Reformer, and wait until the client has an hour or so to sit with us.
Then we let them do our job for us.
We sit and translate their questions into Reformer language (which they could do independently with a bit of training) and get feedback in real-time.

It sounds like a racket, but none of us Redpollers has decades of experience in orthopedics, aircraft maintenance, or plant breeding to call upon. We do not know which questions to ask or which insights are potentially novel.
It would be arrogant to think that any of us laypeople could solve these problems by blindly throwing AI at a dataset.

## Limitations of the humanistic approach

We built Reformer on humanistic AI.
Humanistic AI allows us to chew through analyses.
It is especially powerful for discovery and green-fielding: getting answers from your data when there is no preexisting model or no satisfactory preexisting model.
If you are considering dropping a dataset into a random forest or neural net to see what comes out, your experience would be much improved using a humanistic AI system.
That said, there are situations when you might want to go another route.

The humanistic approach is fast because it is flexible, and it is flexible because it makes few assumptions.
This is how it can go from nothing to a causal model.
A model that makes strong assumptions (that are also correct) will better account for the data.
However, developing that domain-specific model takes a lot of science and iteration.
The humanistic approach can get you 90% of the way there by allowing you to check your modeling assumptions before you start modeling.

# Wrapping up

Data science is a difficult process.
Creating production models takes significant human and machine resources.
Data science is also a wasteful process because one can never know 1) whether what one wants to do is possible or 2) whether the result will be accepted/trusted by the stakeholder.

The primary contributor to data science's difficulty and wastefulness is that Machine Learning structures lend themselves to inefficient modeling approaches.
Machine learning only models specific questions, which places unreasonable demands on the practitioner.
These demands are that the data are well-behaved and complete, and the underlying question is well-formed.
Alternatively, humanistic systems model data, which allows the user flexibility in both the state of the data and the statement &mdash; or existence &mdash; of the question.
The result is the elimination of the vast majority of backtracking and iteration, and the transformation of a months-long process into a weekend project.


# Key points

- The standard data science process, CRISP-DM, is slow and wasteful. It has as many paths away from answers as toward them; and has cycles where the process breaks down.
- CRISP-DM's design was meant to compensate for today's inadequate Machine Learning models.
- Today's Machine Learning is question-oriented rather than data-oriented; therefore one Machine Learning model can address only one question.
- Data-oriented, humanistic AI enables a much simpler workflow in which all paths point toward answers, and the only cycle is answering more questions.
- The humanistic workflow enables _immediate_ ask-and-answer capabilities, enabling stakeholders to directly engage in the discovery process.
- To see how Redpoll's Reformer platform enables faster discovery [click here](@/redpoll-vs-ml.md).
