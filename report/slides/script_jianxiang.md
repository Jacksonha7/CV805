# Talk Script: Jianxiang He (whole study, Q1–Q4)

Deck: `talk_jianxiang.pdf`. Target: **5 minutes** of talk (≈650 words at ~130
words/minute), then 10 minutes of Q&A. Slides 11–15 are backup for questions.
Each slide has a target time, a Chinese reminder (【要点】), and what to say.
**Bold** words are the ones to stress.

---

## Slide 1 · Title (0:10)

【要点】一句话说清题目。

Hi, I'm Jianxiang. Our project asks: when we pick frames from a long video by
mixing two clues, **should the mixing weight change with the question?**

## Slide 2 · Team and my part (0:15)

【要点】先交代分工，突出自己负责整体研究和 Q1。

We are a team of three. I set up the problem and the benchmark, built the data
pipeline, did the Q1 analysis, and tied everything into one study. Longtao led
the continual learning experiments, and Makhmud led the question-answering
evaluation. In this talk I'll give the **whole picture**.

## Slide 3 · Background: two clues, one weight (0:40)

【要点】讲清 VSI 和 λ，用右边的例子。

A vision-language model can only look at about **eight frames** of a long video.
A method called VSI picks them by scoring every second with two clues: whether
the question's objects appear in the frame, and whether the subtitle matches the
question. It mixes the two with a weight lambda and keeps the top eight. VSI
uses **0.3 for every question**.

On the right is a real example. With 0.3, all eight frames **miss** the answer
moment in green. With 0.8, they **hit** it.

## Slide 4 · Our questions and my part (0:30)

【要点】四个研究问题。

So we ask four questions. One: does the best lambda depend on the question type?
Two: can a small predictor learn lambda when question types arrive one after
another, which is called **continual learning**, and does it forget? Three: if
the gain is small, why? Four: do better frames give better answers? We test this
on 753 questions from LongVideoBench, in four question types, with nine runs per
method.

## Slide 5 · Q1: types want different weights (0:35)

【要点】Q1 第一部分：不同题型最优 λ 不同。

For question one: each line shows recall as lambda changes. Text questions are
best at lambda **0.8 to 0.9**; visual questions at **0.3 or lower**. Question by
question, text questions prefer 0.62 and visual ones 0.49, which is highly
significant. But notice that for **more than half** of the questions, lambda
makes no difference at all.

## Slide 6 · Q1: where is the room to improve? (0:35)

【要点】可提升空间分解，大头在逐题差异。

How much can we gain? Just tuning one global value gives **plus 4.5** points.
One value per question type adds only **1.4** more. But the best value for every
single question would add **8.9** more. So the type matters, but most of the
room is in **per-question** differences.

## Slide 7 · Q2–Q3: ties a tuned constant (0:45)

【要点】持续学习比默认好，但只追平调好的常数；原因是对所有题型给出同一个 λ。

Now questions two and three. Every continual learning method beats the default
by **four to five points**, in all nine runs. But against **one tuned value**, the
best method gains only **0.6**. This figure shows why: the average lambda the
predictor gives to each question type. The four lines **overlap** around 0.6. It
learned a better global value, not different values per type. Forgetting is
small for the same reason; LwF halves it, but not consistently.

## Slide 8 · Q4: helps only some questions (0:40)

【要点】下游问答：整体没变，只对部分题有用。

Finally, question four. We gave the frames to Qwen3-VL. Overall, the learned
weight answers **the same** as the default. But frames do matter: when the eight
frames miss every answer moment, accuracy is 0.49; when they hit all, it's 0.75.
And on the 284 questions where lambda actually changes the frames, the best
lambda answers **five points** more.

## Slide 9 · Summary (0:30)

【要点】四句话总结。

To summarize: the best weight depends on the question type, but most of the
gain is per question. Continual learning beats the default but only **ties a
tuned constant**. Better frames help answers only where the weight changes the
frames. And our lesson: always compare a learned weight with a **tuned**
constant. We also found four silent bugs that each reversed our conclusions.

## Slide 10 · Thank you (0:05)

Thank you, I'm happy to take questions.

**Total ≈ 4:45.**

---

## Q&A preparation (10 minutes)

Jump to the backup slide in brackets when it helps.

**Q: What exactly are the four question types?** [Slide 11]
LongVideoBench labels each question with a category. We grouped them into text
perception, visual perception, text relation and visual relation, with 127,
237, 171 and 218 questions.

**Q: What does recall@8 mean?**
Each question has annotated answer moments, about two on average. Recall@8 is
the share of those moments that have at least one of our eight frames within
five seconds.

**Q: Can you show the full results table?** [Slide 12]
Yes. LwF and Joint are best at 0.668. The tuned constant is 0.662, one value per
type 0.677, and the oracle 0.765.

**Q: How is the tuned constant chosen? Is it fair?**
It is chosen on the training split only, by taking the value with the best
average training recall. It is 0.7 in two splits and 0.9 in one. No test data
is used, so it is a fair baseline.

**Q: Why not give the predictor the question type?**
That is our first next step. But even a perfect use of the type only reaches
one value per type, 1.4 points above the tuned constant. The bigger room, about
nine points, needs per-question information.

**Q: What were the silent bugs?** [Slide 13]
Four of them: annotations in frames versus scores in seconds; LongVideoBench's
fixed 24 fps annotation rate; short videos silently dropped; and training on the
average weight but testing on the most likely one. The last one made learned
weights look three to five points worse than the default.

**Q: Is the text-versus-visual difference real or noise?** [Slide 14]
It is significant: p is 1.6 times ten to the minus six on 284 questions. The
histograms barely overlap at the top.

**Q: Do all runs agree?** [Slide 15]
Against the default, every run of every method is better. Against the tuned
constant, they spread around zero.

**Q: Why is forgetting so small?**
Because the predictor gives all types similar weights, so a new type overwrites
little. That is also why continual learning methods differ little here.

**Q: Why does better recall not give better answers?**
Only 38 percent of questions react to lambda, recall is coarse, and the model
often answers from subtitles or common sense. Makhmud's talk covers this in
detail.

**Q: What would you do with more time?**
Use more questions per video, give the predictor the question type, and learn
small per-question corrections on top of one value per type.
