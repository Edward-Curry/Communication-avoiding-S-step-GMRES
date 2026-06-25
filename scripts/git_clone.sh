#!/bin/bash

set -euo pipefail

repository_name="Communication-avoiding-S-step-GMRES"
repository_url="https://github.com/Edward-Curry/Communication-avoiding-S-step-GMRES.git"
repository_directory="${HOME}/${repository_name}"

if [[ "${FRESH_CLONE_HELPER:-0}" != "1" ]]; then
    helper_script="$(mktemp "${TMPDIR:-/tmp}/fresh_clone_seagull.XXXXXX")"
    cp "$0" "$helper_script"
    chmod 700 "$helper_script"
    FRESH_CLONE_HELPER=1 exec bash "$helper_script" "$@"
fi

helper_script="$0"
temporary_clone="${repository_directory}.fresh.$$"

cleanup() {
    rm -rf -- "$temporary_clone"
    rm -f -- "$helper_script"
}

trap cleanup EXIT

if ! command -v git >/dev/null 2>&1; then
    echo "git is not available on this node." >&2
    exit 1
fi

if [[ "${1:-}" != "--yes" ]]; then
    echo "This will replace:"
    echo "  $repository_directory"
    echo
    echo "All uncommitted files, build files, and experiment outputs inside it will be deleted."
    read -r -p "Continue? [y/N] " response

    if [[ "$response" != "y" && "$response" != "Y" ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

cd "$HOME"

echo "Cloning the latest repository from GitHub..."
git clone "$repository_url" "$temporary_clone"

if [[ ! -d "${temporary_clone}/.git" ]]; then
    echo "The fresh clone could not be verified. Existing repository was not changed." >&2
    exit 1
fi

if [[ -e "$repository_directory" ]]; then
    echo "Removing the old repository..."
    rm -rf -- "$repository_directory"
fi

mv -- "$temporary_clone" "$repository_directory"
trap - EXIT
rm -f -- "$helper_script"

echo "Fresh clone ready:"
echo "  $repository_directory"
echo
echo "Next:"
echo "  cd \"$repository_directory\""
echo "  bash scripts/build.sh"
