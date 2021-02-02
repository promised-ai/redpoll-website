+++
title = "Escaping the data science death spiral"
date = 2021-02-05
template = "post.html"
draft = false

[taxonomies]
categories = ["data science", "ai"]

[extra]
author = "Baxter Eaves"
subheading = "How today's machine learning models force us into wasteful processes and how we can escape them"
image = "skull-01.jpg"
theme = "light-transparent"
front_page = false
+++

After five-years in development, multi-million dollar decision support project is patently ignored by decision makers. A nine-figure data-science company lays off 60% of its workforce as it is has failed for the past two years to develop a machine learning model that can help its target market. A hospital scrubs its efforts to integrate an clinical support software after it fails to make useful recommendations, more than three years and $60 million later.

The data science process is hard. It process is inefficient, error-prone, and fragile. It's full of iteration and cycles: we don't like what we're producing, so we take a few steps back, tweak, and try again. Every step we take requires communication with subject area experts and stakeholders; it requires waiting for people to come away from their work to help us not to do the wrong thing. It requires careful archival and documentation of all previous steps to ensure reproducibility, so we continually progress toward the goal and avoid tripping over our past failures. Cycles are fragile. Every time we hit a cycle the process is at risk of breaking. This fragility is why there is so much waste, and why there is so much commercial work supporting the process in the form of infrastructure providers, managed machine learning platforms, and domain-specific machine learning solutions from companies that have navigated the process for us. The process is hard. Ridiculously hard.

I argue that the data science process is hard not because we are not taking enough care &mdash; or even because data science is hard &mdash; but because machine learning is bad; and this difficult, fragile process has been built to accommodate for today's difficult, fragile machine learning.

In this post we will discuss the standard data science process, why it is bad, where it came from, and how we can break free from it.

# The standard data science process: CRISP-DM

If you are a data scientist, you probably use an analytics process called [*Cross-industry standard process for data mining*](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) or CRISP-DM. You might not be aware what you do is called "CRISP-DM", but this is basically the way everybody does analytics because it's basically the only workflow that makes sense with today's machine learning technology. It goes like this:

<style>
    .row {
        padding: 1rem 0;
        align-items: center;
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
        <p>You start by identifying a question that would be valuable to answer</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-02.png" style="max-width: 150px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then you must find data that support answering you question.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-03.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>But it turns out you do not have &mdash; or cannot get &mdash; the data you need, so you must reformulate the question to reflect the state of the data.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-04.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then you choose from one of infinite machine learning models, each with innumerable parameter settings and innumerable potential add-ons and tweaks.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-05.png" style="max-width: 190px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>But the model you chose doesn't support your data, so you have to transform your data. You might have to encode your categorical values as continuous (e.g. one-hot), or transform data that lie across an unfavorable support or that do not adhere to the assumptions of the model (normality?).</p>
        <p>Perhaps you have records with missing values, which you'll either have to fill in or throw out. Imputation requires its own CRISP-DM process and biases your data. Throwing out all reecords with missing cells is wasteful, and for some problems with sparse data you might end up with no records at all.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-06.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then maybe you can't find a model that answers the exact question you want to ask with the data you have. You'll have to ask an adjacent question.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-07.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Once you get the question, data, and first-try model sorted out, you can train the model.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-08.png" style="max-width: 220px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>And you validate/score the model.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-09.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>And it doesn't work (for whatever definition of "work"), because things rarely work on the first try. So you tweak the model or select a different model entirely.</p>
    </div>
</div>

<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-10.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Then one of two things happens: you try a bunch of things and find that you're doing the best you can, or you hit your deadline. In either case, it's time to report. <strong>Congrats! You have answered a question!</strong></p>
    </div>
</div>
<!--
<div class=row>
    <div class="col-xs-12 col-sm col-left">
        <img src="/img/workflows/standard-11.png" style="max-width: 250px;">
    </div>
    <div class="col-xs-12 col-sm col-right">
        <p>Time to do it again.</p>
    </div>
</div>
-->

Things to note:
- People are doing the work; not AI.
- It takes a lot of work to answer a single question.
- Sometimes you don't answer the question because the information doesn't exist in the data.
- There are nearly as many paths back as there are paths forward.
- The question you end up answering is usually not the one you set out to answer.
- Not pictured: all the people you have to bother to get things right.

It's a game of *Chutes and Ladders*. If you're lucky you'll always progress &mdash; up the ladders &mdash; toward an answer. But in this process we, more often than not, find ourselves on the backwards path &mdash; falling down chutes &mdash; having to iterate and try again. In this game of *Chutes and Ladders* there are four chutes and five ladders. Those are not good odds. And each time we retrace the old path, there is a chance we'll fall down those earlier chutes. The harder and more complex your problem the more iteration you'll have to do, and the less the question you end up answering looks like the question you set out to answer.

And here's the thing: in complex domains, like health and biotech, where there is a lot of science involved and high cost for failure, domain experts *must* be involved in these each step of the process (apart from training). Each step requires e-mailing an expert, scheduling time &mdash; days or weeks out &mdash; for them to stop what they're doing, and then sitting down with them so they can tell you what's wrong and what's right. If you're iterating a lot you are either pulling people away from their work a lot, and dragging your work out by months; or you are not pulling people away from their work and are potentially (probably) making dangerous mistakes.

## The Death Spiral

The whole data science cycle is rough, but the Death Spiral is the model - train - validate cycle. This is the cycle where I've seen the most data science projects go off the rails. A lot of things come together to make this part of the process particularly nasty, but most of the trouble is in measuring and maximizing performance.

Machine learning does not give you a good way to know whether your question is answerable with your data or how well you should expect to do. The first time you try a model you fail. Your options are then to find a model that will work or to prove that no model will work. You must succeed, or you must prove unicorns do not exist. Under some circumstances, if you have the math, and the problem is well-behaved, you may be able to mathematically prove that modeling is impossible. But in my experience (much to my dismay), mathematical proof is not compelling to stakeholders. They want empirical results. In practice, the only way to prove that no model will work &mdash; or no model will work better &mdash; is by deduction. To try *everything*.

And there is no shortage of things to try. And since none of them quite fit your question and your data, there is always something to blame. "Well, this model doesn't work well with categorical data, so maybe we could try a different encoding or embedding", or "This deep network only has two layers, so it's probably not very expressive", or "We have a lot of missing data, and we've only tried this one imputation method".

Of course, each time you make a modeling choice you must explain it to, and get the thumbs up from, a subject area expert. Communication becomes a bottleneck. So you either communicate properly and drag your process out by months (or longer), or you do not communicate and deliver something that is likely to fail or to be rejected by decision makers because they don't understand because they haven't been included (hell, they might not understand even if they're included because machine learning is opaque). And it is common for data science projects to be shelved after months or even years of development.

## Why The Data Science Death Spiral Exists

The Data Science Death Spiral exists because machine learning is, and always has been, focused on modeling questions. What is the value of Y given X? Which data are similar? What are the factors that determine Z? Machine learning assumes the user has a well-defined problem and nice, neat, complete data. This is almost never the case. **Your problem is more nuanced than you think and your data are gross**. 

Because machine learning is focused on modeling questions, the standard data science process must focus on modeling questions; so the standard process is good when you have a well-defined problem and nice data, is bad when you have a nebulous problem and ugly data, and is disastrous when doing exploration. **And since the Data Science process is tragically bad for exploration, it is tragically bad for innovation because innovation comes from exploration.**

# Toward a better data science process through humanistic systems

People don't model questions. You don't have to know what you want to learn before you learn it. You go out in the world, you observe data, and learn from those data. You learn about the process that produced the data: the world. What if we had machine learning or artificial intelligence technology that was focused on modeling the whole data rather than just modeling single questions? We'd be able to answer any number of questions within the realm of our data without re-training or re-validating. We'd know which questions we'd be able to answer and how well. The data science process would look like this:

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
        <p>Ask a quesiton.</p>
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

The first thing to notice is that there are far fewer steps and that every step takes us closer to where we want to be. The key in this process is asking sensible questions. Asking sensible questions requires deep domain expertise, so it would be most effective for this Data Science process to be run by the domain expert.

Eliminating iteration through up-front learning and turning a *question-modeling* system into a *question-answering* system dramatically reduces discovery and production time. Under this process, discovery-to-production takes hours or days rather than months or years.

**And the best part is this technology exists today. You can check out what this looks like in practice <span style="color: crimson">[here](@/redpoll-vs-ml.md)</span> and <span style="color: crimson">[here](@/blog/gamma.md)</span>.**

When we consult for clients, we take their data, feed it through Reformer, then we wait until the client has an hour or so to sit with us. Then we let them do our job for us. We just sit and translate their questions into Reformer language (which they could do with a bit of training) and get feedback in real time. It sounds like a racket, but I don't have 20 years of orthopedics, or aircraft maintenance, or plant breeding experience to call upon, so I do not know which questions to ask or which insights are interesting. It would be arrogant to think that I, a layperson, could solve these problems by blinding throwing AI at data.

## Limitations of the humanistic approach

Reformer is built on humanistic AI. Humanistic AI allows us to chew through analyses. It's especially powerful for discovery and green-fielding: getting answers from your data when there is no preexisting model or no satisfactory preexisting model. If you're considering dropping a dataset into a random forest or neural net to see what comes out, you would be better of using a humanistic AI system. That said, there are situations when you might want to go another route.

The humanistic approach is fast because it is flexible, and it is flexible because it makes few assumptions. This is how it can go from nothing to causal model. A model that makes strong assumptions (that are also correct) will better account for the data. This is the [no free lunch theorem](https://en.wikipedia.org/wiki/No_free_lunch_in_search_and_optimization). However, developing that strong model takes a lot of science and iteration, and the humanistic approach can get you 90% of the way there by allowing you to check your modeling assumptions before you start modeling.

# Wrapping up

Data science is a difficult process. Creating production models takes a lot of effort from a lot of people and a lot of IT resources. Data science is also a wasteful process because you can never know 1) whether what you want to do is possible, or 2) whether the result will be accepted/trusted by the stakeholder.

The primary contributor to the difficulty and wastefulness of data science is that machine learning takes the wrong modeling approach. Machine learning models questions, which places unreasonable demands at the practitioner, namely that the model be provided with well-behaved, complete data, and that the question be well-formed. Alternatively, humanistic systems model data, which allows the user flexibility in both the state of the data and the state &mdash; or existence &mdash; of the question. The result is the elimination of the vast majority of backtracking and iteration.


# Key points

- The standard data science process, CRISP-DM, is slow and wasteful. It has as many paths away from answers away from answers as toward them; and has cycles where the process breaks down.
- CRISP-DM is designed to compensate for today's inadequate machine learning models.
- Today's machine learning is question-oriented rather than data-oriented, therefor one machine learning model can answer one question.
- Data-oriented humanistic AI enables a much simpler workflow in which all paths point toward answers and the only cycles is answring more questions.
- The humanistic workflow allows users to ask and answer questions immediately and enables stakeholders to engage in the discovery process directly.
- To see how Redpoll's Reformer platform enables faster discovery [click here](@/redpoll-vs-ml.md)[here].
