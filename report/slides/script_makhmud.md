# Talk Script: Makhmud Assentay (question answering, Q4)

Deck: `talk_makhmud.pdf`. Target: **5 minutes** of talk (≈650 words at ~130
words/minute), then 10 minutes of Q&A. Slides 11–14 are backup for questions.
**Bold** words are the ones to stress.

---

## Slide 1 · Title (0:10)

Hi, I'm Makhmud. Our project studies how to pick frames from long videos. My
part answers the final question: **do better frames actually give better
answers?**

## Slide 2 · Team and my part (0:15)

We are a team of three. Jianxiang set up the problem and led the overall study.
Longtao ran the continual learning experiments. I built and ran the
**question-answering evaluation**, which is question four of our study.

## Slide 3 · Background: two clues, one weight (0:40)

A vision-language model can only see about **eight frames** of a long video. A
method called VSI picks them using two clues for every second: whether the
question's objects appear, and whether the subtitle matches the question. It
mixes them with a weight lambda and keeps the top eight. VSI uses **0.3 for
every question**. In this real example, 0.3 misses the answer moment, but 0.8
hits it.

## Slide 4 · My part: does it help answers? (0:30)

My teammates changed lambda to pick frames that hit the answer moment more
often. That is measured by **recall at eight**. But the real goal is to
**answer the question**. Recall is only a stand-in. So my job was to check: when
recall goes up, does answer accuracy go up too?

## Slide 5 · How we test answers (0:40)

We use **Qwen3-VL-8B**. For each question, we give it the eight chosen frames,
each with the subtitle shown at that moment, followed by the multiple-choice
question, and ask for one letter. We compare many frame choices: even spacing,
fixed weights from zero to one, the best weight per question, and the learned
predictors. The predictors are tested only on their own test questions and
compared with the default on **exactly the same questions**. When two methods
pick the same frames, we call the model once, so the whole study took about
3,500 calls.

## Slide 6 · Result: same accuracy as default (0:45)

Here are the results. The default weight 0.3 gets 63.1 percent. The learned
predictor is only **0.4 points** better, and that is not significant. Even the
best possible weight per question adds only **1.8 points**. Interestingly,
lambda 0.7, which has the best recall, answers slightly **worse** than 0.3. And
using both clues beats using only the picture or only the subtitles.

## Slide 7 · Frames matter, for some questions (0:50)

So do frames not matter? They do. On the left, at lambda 0.3: when the eight
frames hit **no** answer moment, accuracy is **0.49**. When they hit **all** of
them, it is **0.75**. So hitting the right moment helps a lot.

On the right, I split the questions into two groups. For 469 questions, lambda
does not change the selected frames at all, so every method gives the same
answers. But for the **284 questions** where lambda does change the frames, the
best weight answers **five points** more questions correctly.

## Slide 8 · Why recall is not accuracy (0:40)

So why does the overall number barely move? Three reasons. First, only **38
percent** of questions react to lambda at all. Second, recall is coarse: each
question has only about two answer moments. Third, the model often answers from
subtitles or common sense; even evenly spaced frames reach 61.5 percent.

On the 284 sensitive questions, our learned predictor gains **1.5 points**, but
the confidence interval still includes zero. It captures only a small part of
the five-point room.

## Slide 9 · Takeaways from Q4 (0:25)

To summarize: overall, the learned weight answers the same as the default. Frames
still matter: hitting the answer moment raises accuracy from 0.49 to 0.75. The
weight matters only for the questions where it changes the frames. And the
lesson: report **answer accuracy**, not only recall, and look at the questions
where the change actually applies.

## Slide 10 · Thank you (0:05)

Thank you. I'm happy to take questions.

**Total ≈ 5:00.**

---

## Q&A preparation (10 minutes)

Jump to the backup slide in brackets when it helps.

**Q: Which question types benefit most?** [Slide 11]
Text perception benefits most: 76.4 percent at the default, 79.5 percent with
the best weight. Visual perception does not change at all.

**Q: How do you know the +0.4 is not significant?** [Slide 12]
We compare on the same questions. Out of 687 test questions, 24 were right only
with the learned weight and 21 only with the default. A McNemar test on these
gives p = 0.77, and the bootstrap 95 percent interval is −1.5 to +2.3 points.

**Q: Why use Qwen3-VL-8B and not a bigger model?**
It is a strong open model that fits on one GPU. A stronger model might rely even
more on subtitles and common sense, so frame choice might matter even less. That
would be interesting to test.

**Q: Why include subtitles in the prompt?**
LongVideoBench is designed with interleaved subtitles, and many questions ask
about what is said. Without subtitles, text questions would be much harder for
every method.

**Q: How did you match frames and subtitles?**
For each selected second, we add the subtitle lines within one second of it.
For clipped videos we shift subtitle times by the clip start.

**Q: Did you find any problems while building the evaluation?** [Slide 13]
Yes. While matching frames to answer moments, we found that LongVideoBench
counts annotated frames at a fixed 24 frames per second, not at the video's real
30. On 39 questions whose answer appears in a subtitle, 38 line up at 24 fps but
only 14 at the real rate. After fixing it, the team reran all experiments.

**Q: Why does lambda 0.7 have better recall but lower accuracy?** [Slide 14]
0.7 helps text questions a lot but hurts visual questions slightly. For
answering, losing visual frames on visual questions costs more than the extra
recall gains on text questions, where subtitles already help.

**Q: Is 753 questions enough?**
It limits statistical power: each predictor is tested on 229 questions per
split. That is why differences of one or two points are not significant. More
questions per video would help.

**Q: What would you change in the evaluation?**
Use more questions and a second VLM, and report results separately for the
questions where lambda changes the frames, since that is where any effect
appears.
