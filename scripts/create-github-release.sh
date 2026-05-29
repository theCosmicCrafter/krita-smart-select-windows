#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
  cat >&2 <<'EOF'
Usage: scripts/create-github-release.sh <version>

Example:
  scripts/create-github-release.sh 0.1.0

Environment:
  ALLOW_DIRTY=1  allow releasing from a dirty worktree
  DRAFT=1        create the GitHub release as a draft
EOF
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required: https://cli.github.com/" >&2
  exit 2
fi

cd "$ROOT_DIR"

if [[ "${ALLOW_DIRTY:-0}" != "1" ]] && [[ -n "$(git status --porcelain)" ]]; then
  cat >&2 <<'EOF'
Working tree is dirty. Commit or stash changes before creating a release.

Use ALLOW_DIRTY=1 only for local test releases.
EOF
  exit 2
fi

RELEASE_VERSION="${VERSION#v}"
TAG="v$RELEASE_VERSION"
ZIP_PATH="$ROOT_DIR/dist/krita-smart-select-linux-x64-$RELEASE_VERSION.zip"
TARBALL_PATH="$ROOT_DIR/dist/krita-smart-select-linux-x64-$RELEASE_VERSION.tar.gz"

"$ROOT_DIR/scripts/package-linux.sh" "$RELEASE_VERSION" >/dev/null

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  git tag -a "$TAG" -m "Release $TAG"
fi

if ! git ls-remote --exit-code --tags origin "$TAG" >/dev/null 2>&1; then
  git push origin "$TAG"
fi

release_args=(
  "$TAG"
  "$ZIP_PATH"
  "$TARBALL_PATH"
  --title "$TAG"
  --generate-notes
)

if [[ "${DRAFT:-0}" == "1" ]]; then
  release_args+=(--draft)
fi

if gh release view "$TAG" >/dev/null 2>&1; then
  gh release upload "$TAG" "$ZIP_PATH" "$TARBALL_PATH" --clobber
else
  gh release create "${release_args[@]}"
fi

echo "Published GitHub release: $TAG"
echo "Installable ZIP: $ZIP_PATH"
