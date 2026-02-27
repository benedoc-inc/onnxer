# Git Hooks

Hooks auto-configure when they run. Run `make setup` once after cloning.

## Hooks

- **pre-commit**: Fast checks (<5s) - auto-formatting, compilation, vet, conflict/key/binary detection
- **pre-push**: Comprehensive checks - tests, linting, branch protection, large file detection
- **post-checkout**: Auto-configure hooks on checkout
- **post-merge**: Auto-configure hooks on merge/pull

## Skipping Hooks

- Pre-commit: `git commit --no-verify`
- Pre-push: `git push --no-verify`

## Environment Variables

### Pre-push Hook

- `VERBOSE=true` - Show detailed output
- `ALLOW_DIRTY_PUSH=true` - Allow push with uncommitted changes
- `ALLOW_PUSH_TO_MAIN=true` - Allow direct push to main branch
- `SKIP_ENV_CHECKS=true` - Skip environment verification
- `SKIP_LINTS=true` - Skip linting checks
- `SKIP_TESTS=true` - Skip test execution
