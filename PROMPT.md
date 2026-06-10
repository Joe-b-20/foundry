# mathlab — research goal

I want a small neural network that, given enough math problems, discovers its
own internal algorithms for solving them — and that I can extract those
algorithms from to inspect what it found.

The interesting part is the *discovery*. I don't want to teach it how humans
do math. I want to see what procedures it converges on when it's free to find
its own.

The simplest possible version of the goal:

- Give the model `(a, b) -> a + b` examples and let it figure out addition.
- Then mix in `(a, b) -> a - b` examples. See if it discovers that subtraction
  is addition + negation, or finds something else, or fails entirely.
- Then multiplication. Then division. Each one is a real test of whether
  whatever it learned generalizes.

That's the seed. The far goal is: train it long enough on enough operations
that it finds procedures humans haven't found. I do not expect that part to
work easily, but the project is structured so that each stage produces
something useful even if the moonshot doesn't materialize.

## What I want you to do

You have full freedom over architecture, training method, observability,
file structure, and how many experiments to run. Try many things in parallel
if you can. Branch, abandon, restart. Spawn subagents if useful.

Two things I want you to actively resist:

1. **Don't default to what the literature says works.** I am not trying to
   reproduce published work. I want approaches that exploit the fact that
   this is small, narrow, and from-scratch. Weird ideas are welcome. If you
   find yourself reaching for "small transformer with REINFORCE because that's
   what AlphaCode-style work does," stop and ask whether there's a stranger
   approach that fits the specific constraints here.

2. **Don't keep iterating on stuck approaches.** If an experiment isn't
   producing signal after a reasonable attempt, abandon it. Mark it dead in
   the tracker. Move to a different idea. The point is to explore many
   directions, not to make any single one work through brute force.

## Operating rules

Read `.claude/RULES.md` for the working conventions. The most important one:
**TRACKER.md must be updated after every experiment**, including the failed
ones. Especially the failed ones. You're going to be in this folder for a
long time and you cannot retry things you've already failed at.

I will check in periodically by reading TRACKER.md. I should be able to tell
from it whether genuine exploration is happening.

## Constraints

- This runs on my hardware (RTX 4060 + CPU). Don't write training loops that
  assume H100 scale.
- Conda env named `mathlab`. Set it up if not present.
- Python 3.10+.
- Math eval is exact. `2 + 3` is `5`, not `5.0001`. No partial credit on
  arithmetic correctness.

Begin by reading `.claude/RULES.md`, setting up the environment, and writing
the first entry in `TRACKER.md` (your initial plan and what assumptions
you're making). Then start exploring.