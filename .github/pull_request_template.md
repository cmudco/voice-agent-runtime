<!-- Read .github/CONTRIBUTING.md first. Write "none" in any section that does not apply. -->

## What

<!-- One paragraph. -->

## Why

<!-- The problem this solves. -->

Closes #

## How

<!-- The approach, and any decision that is not obvious. -->

## Testing

<!-- What you ran, and what you checked by hand. -->

## Checklist

- [ ] Branched from `main` and targets `main`
- [ ] `python -m compileall -q runtime scripts tests` passes
- [ ] `python -m unittest discover -s tests` passes
- [ ] Manual voice checks from docs/how-to/testing.md are run, if the pipeline or browser events changed
- [ ] No transcript text, audio or tokens are logged by default
- [ ] New or changed environment variables are in `.env.example` and `docs/reference/configuration.md`
- [ ] `AGENTS.md` or a skill in `.agents/skills/` is updated, if a command or trap changed
- [ ] PRs in other repos are linked, with the deploy order

## AI assistance

<!-- Did an AI coding agent write part of this? Name it, and say what you verified by hand. -->
