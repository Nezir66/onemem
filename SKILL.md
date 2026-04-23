---
name: disciplined-coding
description: Use before writing, editing, or refactoring ANY code — enforces four disciplines (Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution) to prevent over-engineering, scope creep, silent assumptions, and untestable "make it work" tasks. Trigger whenever the user asks to add a feature, fix a bug, refactor, implement, change, or improve code, even if the request seems simple.
---

# Disciplined Coding

Four disciplines to apply before and during any coding work. Each exists because skipping it produces predictable failure modes: hidden assumptions that break later, bloated code that's hard to maintain, unrelated edits that leak into diffs, and tasks that drift because "done" was never defined.

Apply all four. They reinforce each other.

## 1. Think Before Coding

Don't assume. Don't hide confusion. Surface tradeoffs.

Before implementing:

- **State your assumptions explicitly.** If uncertain, ask.
- **If multiple interpretations exist, present them** — don't pick silently.
- **If a simpler approach exists, say so.** Push back when warranted.
- **If something is unclear, stop.** Name what's confusing. Ask.

**Why this matters:** Silent assumptions compound. A wrong interpretation caught before coding costs a sentence; caught after coding costs a rewrite. The user can only correct what you make visible.

## 2. Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

**The senior-engineer test:** Ask yourself — "Would a senior engineer say this is overcomplicated?" If yes, simplify.

**Why this matters:** Every line is a liability — it must be read, understood, and maintained. Speculative flexibility almost never matches the real future need, and leaves behind scaffolding that obscures the actual logic.

## 3. Surgical Changes

Touch only what you must. Clean up only your own mess.

**When editing existing code:**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, **mention it** — don't delete it.

**When your changes create orphans:**

- Remove imports/variables/functions that _your_ changes made unused.
- Don't remove pre-existing dead code unless asked.

**The test:** Every changed line should trace directly to the user's request. If a line in your diff can't be justified by the task, it doesn't belong.

**Why this matters:** Mixed-purpose diffs are hard to review, hard to revert, and hide bugs. Users lose trust when "fix X" also silently rewrites Y. Staying surgical keeps changes reviewable and reversible.

## 4. Goal-Driven Execution

Define success criteria. Loop until verified.

Transform vague tasks into verifiable goals:

- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**Why this matters:** Strong success criteria let you loop independently — you know when you're done. Weak criteria ("make it work") require constant clarification and tend to produce code that compiles but doesn't actually solve the problem.

## How these fit together

- **Think** prevents wasted work.
- **Simplicity** prevents wasted code.
- **Surgical** prevents wasted scope.
- **Goal-driven** prevents wasted loops.

If you catch yourself violating one, the others are usually slipping too. Stop, re-read this skill, and restart the task from the "Think" step.
