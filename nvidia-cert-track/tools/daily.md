# The Daily Loop (~2 h)

Same shape every study day. Consistency beats intensity — 5 honest 2-hour days outperform
one weekend binge, and Anki only works if it runs daily.

## 1. Read — 45 min

- Open the current week's `plan.md`, do today's line. Read the listed resource(s) only —
  no rabbit-holing; unrelated-but-interesting links go into a "later" list at the bottom
  of your `notes.md`.
- Read actively: after each section, close the tab and add 2–3 bullets to `notes.md`
  from memory, then check yourself. If you can't reconstruct it, re-read that section
  once — not the whole page.
- Pre-sales lens: for every technology, force one sentence of "when would I recommend
  this to a customer, and against what alternative?" That framing is also exactly what
  scenario-style exam questions test.

## 2. Lab / practice — 45–60 min

- **Lab days** (plan.md says so): run the lab. Time-box it — if you're stuck > 20 min on
  environment issues, note the blocker in `notes.md`, kill the instance, and fall back to
  reading the lab's expected-output sections. A half-run lab you understood beats a
  perfect environment you fought all evening. **Always terminate cloud instances before
  closing the laptop** — set a billing alert on day one.
- **Non-lab days**: do 10–15 questions from the week's `self-check.md` (or a previous
  week's, closed-book) OR hand-draw the day's architecture from memory (DGX topology,
  GPU Operator stack, Slurm daemons…) and compare against the docs.
- Month 3 rule: labs take priority over reading. The NCP-AIO exam has a hands-on section;
  fingers-on-keyboard time is the scarce resource.

## 3. Flashcards + notes — 15 min

- Anki: clear today's due cards for the current month's deck. If reviews exceed ~15 min,
  don't skip — cut new cards/day in the deck options instead.
- Close out `notes.md` for today: one line "hardest thing today: ___". Friday-you uses
  these lines to pick what to re-drill.

## Friday variant

Replace the lab slot with: full `self-check.md` closed-book → score it → tick exit
criteria in `plan.md` → fill in this week's row in `PROGRESS.md`. If any exit criterion
fails, schedule it into Monday's read slot explicitly.

## Review-week variant (weeks 4, 8, 12)

- Days 1–2: weakest-domain drilling (from PROGRESS.md confidence scores).
- Day 3: mock exam, real conditions — timed, closed-book, one sitting.
- Day 4: mark it, then re-study *only* the missed questions' domains.
- Day 5: light review + logistics (ID, proctoring check per `booking-checklist.md`).
  No new material within 48 h of the exam.
