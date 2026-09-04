# Contributing

## Ground rules

1. **No proprietary material.** No imagery, labels, sensor specifications, requirement
   documents or company names from any non-public source. Public datasets only. This is a
   hard constraint on every commit.
2. **No experiment defined in a script.** If a run cannot be reproduced from
   `configs/experiment/*.yaml`, it does not count.
3. **Every reported number carries a seed spread.** Single-seed results are not reported.
4. **A negative result is a result.** If RFS fails to predict downstream utility, that
   finding is reported as clearly as a positive one would be.

## Workflow

```bash
make setup
make lint
make test
```

Pre-commit runs ruff and blocks large files. CI runs lint and tests on every push.
