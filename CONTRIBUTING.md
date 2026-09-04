# Contributing

## Ground rules

1. **No proprietary material.** No imagery, labels, sensor specifications, requirement
   documents, internal results, or organisation names from any non-public source. Public
   datasets only. This is a hard constraint on every commit, and it is not satisfied by
   paraphrasing: a specific figure, threshold or conclusion taken from a non-public document
   is proprietary whether or not the document is named.
2. **Every claim about prior work carries a citation.** Statements of the form "X is widely
   reported" belong in `docs/references.md` with a source, or they do not belong here. If a
   result cannot be traced to a public source, it is not cited, not reproduced, and not used
   as a baseline.
3. **No experiment defined in a script.** If a run cannot be reproduced from
   `configs/experiment/*.yaml`, it does not count.
4. **Every reported number carries a seed spread.** Single-seed results are not reported.
5. **A negative result is a result.** If RFS fails to predict downstream utility, that
   finding is reported as clearly as a positive one would be.

## Workflow

```bash
make setup
make lint
make test
```

Pre-commit runs ruff and blocks large files. CI runs lint and tests on every push.
