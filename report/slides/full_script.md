# Talk Script: Should the Fusion Weight Change with the Question?

Matches `report/slides/full_talk.pdf` (22 slides). About 12 minutes at a relaxed pace
(≈130 words per minute). Each slide has a target time, a one-line Chinese
reminder of its purpose (【要点】), and what to say. Words in **bold** are the
ones to stress. Slides 20–22 are backup, only for questions.

---

## Slide 1 · Title (0:15)

【要点】自我介绍，一句话说清题目。

Hi everyone, I'm Jianxiang. My project asks a simple question: when we pick
frames from a long video by mixing two kinds of clues, **should the mixing weight
change with the question?**

## Slide 2 · Outline (0:15)

【要点】快速过一遍结构，不要停留。

I'll first explain the problem and our setup, then go through four research
questions, and finish with some lessons from the project.

## Slide 3 · Long videos, few frames (0:40)

【要点】讲清楚为什么要选帧。

A video can be tens of minutes long, which is thousands of frames. But a
vision-language model can only look at around **eight frames** at a time. If
those eight frames miss the moment the question is about, the model simply
cannot answer. So before we call the model, we need to find the frames that
matter. This step is called **keyframe search**.

## Slide 4 · VSI: two clues, one weight (0:50)

【要点】讲清楚 VSI 的两个分数和 λ 是什么。

Our starting point is a method called VSI. For every second of the video, VSI
computes two scores. The **visual score** checks whether the objects mentioned in
the question appear in the frame. The **subtitle score** checks whether the
subtitle at that moment matches the question. VSI adds the two with a weight
lambda and keeps the top eight seconds. Lambda zero means picture only, lambda
one means subtitles only, and VSI uses **0.3 for every question**.

But think about two questions: "what does the subtitle say when…" and "what
color is the man's shirt". The first should trust subtitles more, the second
should trust the picture more. So is one fixed weight really right for all
questions?

## Slide 5 · One real example (0:40)

【要点】用真实例子让听众"看见"λ 的作用。

Here is a real question from LongVideoBench. It mentions a subtitle, "Just pour
the…". The orange line is the visual score, the blue line is the subtitle score,
and the green band is where the answer actually is. With the default weight 0.3,
the eight selected frames, shown in grey, **all miss** that moment. If we change
lambda to 0.8, most of the frames land **right on it**. So the weight can decide
whether the model sees the answer at all.

## Slide 6 · Our idea: learn the weight (0:45)

【要点】引出持续学习和四个研究问题。

So our idea is to train a small predictor that looks at each question and
outputs its own lambda. In practice, new types of questions keep appearing, so
the predictor should learn them **one after another**, without forgetting the
old ones. This setting is called **continual learning**.

We ask four questions. One: does the best weight really depend on the question
type? Two: can continual learning learn it, and does it forget? Three: if it
doesn't work well, what is holding it back? And four: do better frames actually
give better answers?

## Slide 7 · Experimental setup (0:50)

【要点】一页讲完设置，重点强调"调好的常数"这个基线。

We use 753 questions from LongVideoBench and group them into four question
types: text perception, visual perception, text relation, and visual relation.
The predictor learns them in order. We try three different orders and three
random splits, so every method is run **nine times**.

We compare six methods: naive training, three classic continual learning
methods called EWC, LwF and replay, a nearest-neighbor method, and joint
training on everything as a reference.

The most important part is the baselines. Besides the default 0.3, we also use
**one lambda tuned on the training set**. This turns out to matter a lot. We
also report one lambda per question type, and an oracle that picks the best
lambda for every question, as an upper limit.

Our main metric is recall at eight: how many answer moments the eight frames
hit.

## Slide 8 · Q1: types want different weights (0:45)

【要点】回答 Q1 的第一部分：不同题型最优 λ 不同。

Let's start with question one. Each line shows how recall changes with lambda
for one question type. The two text types, in blue, are best at lambda 0.8 to
0.9. The two visual types, in orange, are best at 0.3 or lower. For text
perception, moving lambda from 0.3 to 0.8 raises recall from **0.73 to 0.88**.

But look at the right panel: for **more than half** of the questions, lambda
makes no difference at all. Keep this in mind, it comes back later.

## Slide 9 · Q1: question by question (0:30)

【要点】逐题统计也显著，Q1 的答案是"是"。

If we look question by question, using only the questions where lambda matters,
text questions prefer a mean lambda of 0.62 and visual questions 0.49. The
difference is highly significant. So the answer to question one is **yes**: the
best weight depends on the question type.

## Slide 10 · Q1: where is the room to improve? (0:45)

【要点】可提升空间分三段，大头在逐题差异。

How much can we gain, and where? Starting from the default 0.3, simply tuning
one global value gives **plus 4.5** points. Allowing one value per question type
adds only **1.4** more. But choosing the best value for every single question
would add another **8.9**. So most of the room is in **per-question differences**,
and only a minority of questions carry any signal about them.

## Slide 11 · Q2: main results (0:50)

【要点】结果比默认好，但只追平调好的常数。

Now question two: what does continual learning actually achieve? All methods
beat the default weight by four to five points, in **all nine runs**. That looks
great at first. But compared with one lambda tuned on the training set, the best
methods, LwF and joint training, gain only **0.6** points. And none of them
reaches one lambda per question type.

On forgetting, LwF and joint forget about half as much as naive training, but
this only holds in six of the nine runs, so the evidence is weak.

## Slide 12 · Q2: run by run (0:30)

【要点】强调选对基线的重要性。

This slide shows every run. On the left, against the default, every dot is
above zero. On the right, against the tuned value, they sit **around zero**. The
lesson is that a learned weight should be compared with a **tuned constant**, not
with the default from the original paper. Otherwise we overstate how much
learning helps.

## Slide 13 · Q3: one weight for every type (0:45)

【要点】核心发现：预测头对所有题型给出几乎相同的 λ。

So why is the gain so small? This figure shows the average lambda the predictor
gives to each question type after each training stage. The triangles on the
right are the best values for each type. The four lines **almost overlap** at
about 0.6. The predictor did not learn to give text questions a higher weight
and visual questions a lower one. It just learned a **better global value**. That
is why it only ties the tuned constant.

## Slide 14 · Q3: why no differences? (0:45)

【要点】两个原因，加上为什么遗忘小。

We see two reasons. First, the training signal is **sparse**: 62 percent of the
questions give the same result for any lambda, so almost all learning comes
from the remaining third. Second, the predictor has **no information about the
question type**. It only sees the two score curves, and these look similar across
types. We also tried adding the question text, but with only 753 questions it
overfit and got worse.

This also explains why forgetting is small: all types end up with similar
weights, so learning a new type does not overwrite much.

## Slide 15 · Q4: answering with Qwen3-VL-8B (0:45)

【要点】下游问答：基本没有提升。

Finally, question four: do better frames give better answers? We gave the
selected frames and their subtitles to Qwen3-VL and measured accuracy. The
learned weight and the default give **the same accuracy**. Even the oracle
weight adds only 1.8 points. And interestingly, lambda 0.7, which has the best
recall, answers slightly worse than 0.3. So better recall does not automatically
mean better answers.

## Slide 16 · Q4: it matters for some questions (0:45)

【要点】λ 只对一部分题有用，在这部分题上空间有 5 个点。

But that doesn't mean frames don't matter. On the left: when the eight frames
miss every answer moment, accuracy is 0.49; when they hit all of them, it's
0.75. On the right, we split the questions. On the questions where lambda does
not change the frames, nothing changes, as expected. On the **284 questions**
where it does, the oracle weight answers **five points** more questions
correctly. So the weight does matter, but only for about a third of the
questions, and our predictor captures only a small part of that.

## Slide 17 · Four mistakes we fixed (0:45)

【要点】工程教训，说明结论来之不易。

I also want to share four mistakes we found along the way. Each of them ran
without any error message, but gave wrong results. The annotations were in
frames while our scores were in seconds. LongVideoBench counts frames at a
fixed 24 frames per second, not the real 30. Short videos were silently dropped.
And the most dangerous one: we trained on the average lambda but tested on the
most likely one. Before we fixed that, learned weights looked **three to five
points worse** than the default, and we would have concluded that learning
doesn't work at all.

## Slide 18 · Summary (0:45)

【要点】三句话总结 + 下一步。

To summarize. First, the best weight **does** depend on the question type.
Second, continual learning beats the default, but only **ties a tuned constant**,
because it collapses to one weight for all types. Third, better frames help
answers only for the questions where the weight changes the frames. The next
step is to predict per-question differences, for example by giving the
predictor the question type, or by starting from one weight per type and
learning small corrections.

## Slide 19 · Thank you (0:10)

Thank you. I'm happy to take questions.

---

## Backup slides (only if asked)

- **Slide 20 · Annotation frame rate**: use when asked how we know the
  annotations are at 24 fps. "We checked 39 questions whose answer appears in a
  subtitle. At 24 fps, 38 of them line up with the subtitle within five seconds;
  at the real frame rate only 14 do."
- **Slide 21 · Train/test mismatch**: use when asked about the decoding bug.
  "The predictor spread its probability between lambda zero and one. The average
  was a sensible middle value, but the single most likely value was one of the
  two worst extremes."
- **Slide 22 · Predictor inputs**: use when asked why not use the question text.
  "The question embedding has 768 dimensions and we only have about 500
  training questions, so it overfits. Curve statistics plus our 44 hand-designed
  features work best."

---

## Likely questions and suggested answers

**Q: Why not just give the predictor the question type?**
That's a fair point and it's our first next step. With the type as input, the
best it could do is one weight per type, which is only 1.4 points above the
tuned constant. The bigger gain, about nine points, needs per-question
information.

**Q: Isn't 753 questions too small?**
Yes, it's a limitation. Each test split has about 229 questions, which is why
the question-answering differences are not significant. We used one question
per video because the scores are computed per video.

**Q: Why is forgetting so small? Is continual learning even needed here?**
Because the predictor gives all types similar weights, there is little conflict
between tasks. With a single number as output, continual learning matters less
than predicting per-question differences. That is one of our conclusions.

**Q: Why does better recall not give better answers?**
Recall@8 is coarse: each question has about two annotated moments, so recall
only takes a few values. Also, the model can answer many questions from the
subtitles or general knowledge even without the exact frame; even evenly spaced
frames reach 61.5 percent.

**Q: Why min-max scaling and not Z-score?**
We follow what the VSI code actually does, which is min-max scaling, so our
default baseline matches the original method.

**Q: How did you choose 0.7 for the tuned constant?**
It is chosen on the training split only, by taking the value with the best
average training recall. It is 0.7 in two of the three splits and 0.9 in the
third. No test data is used.
