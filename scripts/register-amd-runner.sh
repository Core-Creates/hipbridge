#!/usr/bin/env bash
# Register this AMD box as a self-hosted GitHub runner for hipbridge.
#
# Exists because every hardware number in this repository was produced by a
# person on an SSH session to a rented box, and that box went down repeatedly
# mid-measurement. Nothing verifies this project on a device unless someone is
# watching, which means a kernel regression reaches main and stays there.
#
#   bash scripts/register-amd-runner.sh <repo-url> <registration-token> [labels]
#
# The token comes from  Settings -> Actions -> Runners -> New self-hosted runner
# and expires in about an hour. It is a registration token, not a PAT: it can
# add a runner to one repository and nothing else. Do not substitute a personal
# access token here, and especially not the broadly scoped one this project's
# droplet has been carrying.
#
# The runner is installed under ~/actions-runner and runs in the foreground so
# it dies with the box, which for an ephemeral rented GPU is the honest
# lifetime. Use --once semantics via `run.sh --once` if you want it to take one
# job and exit.
set -eu

URL="${1:-}"
TOKEN="${2:-}"
LABELS="${3:-self-hosted,amd,gfx942}"

if [ -z "$URL" ] || [ -z "$TOKEN" ]; then
    echo "usage: $0 <repo-url> <registration-token> [labels]" >&2
    echo "e.g.   $0 https://github.com/Core-Creates/hipbridge ABC123..." >&2
    exit 2
fi

say() { printf '\n== %s\n' "$*"; }

say "What this box actually has"
command -v hipcc >/dev/null 2>&1 && hipcc --version | head -2 || echo "hipcc: NOT PRESENT"
command -v rocminfo >/dev/null 2>&1 && rocminfo 2>/dev/null | grep -m1 gfx || echo "rocminfo: NOT PRESENT"
if ! command -v hipcc >/dev/null 2>&1; then
    echo
    echo "No hipcc, so a runner here would report SKIP for every suite and prove"
    echo "nothing. Fix the toolchain first: bash scripts/install-hip-headers.sh"
    exit 3
fi

say "Fetching the runner"
mkdir -p "$HOME/actions-runner"
cd "$HOME/actions-runner"
if [ ! -x ./config.sh ]; then
    VERSION="$(curl -fsSL https://api.github.com/repos/actions/runner/releases/latest |
        sed -n 's/.*"tag_name": *"v\([^"]*\)".*/\1/p' | head -1)"
    [ -n "$VERSION" ] || { echo "could not determine the runner version" >&2; exit 4; }
    echo "runner $VERSION"
    curl -fsSL -o runner.tar.gz \
        "https://github.com/actions/runner/releases/download/v${VERSION}/actions-runner-linux-x64-${VERSION}.tar.gz"
    tar xzf runner.tar.gz
    rm -f runner.tar.gz
fi

say "Registering"
# The runner refuses to configure as root, which is a reasonable guard: CI jobs
# execute arbitrary workflow code from the repository, and doing that as root is
# a poor trade on a machine holding anything worth stealing.
#
# Rented GPU boxes are handed to you as root, with ROCm, the repo and the venv
# all set up there, so a dedicated user means re-doing that and adding it to the
# render and video groups for device access. RUNNER_ALLOW_RUNASROOT is the
# official escape hatch and is set here deliberately, on the understanding that
# this box is disposable. Do not do this on a machine you care about, and note
# that a box holding a broadly scoped token is one you care about.
if [ "$(id -u)" = "0" ]; then
    echo "running as root: setting RUNNER_ALLOW_RUNASROOT=1 (see the comment above)"
    export RUNNER_ALLOW_RUNASROOT=1
fi
# --unattended so this never blocks on a prompt, --replace so re-running after
# the box is rebuilt does not accumulate dead runners in the repository's list.
./config.sh --unattended --replace \
    --url "$URL" \
    --token "$TOKEN" \
    --labels "$LABELS" \
    --name "$(hostname)-amd" \
    --work _work

say "Running"
echo "Ctrl-C stops it. The runner appears under Settings -> Actions -> Runners."
echo "Then: Actions -> AMD verify -> Run workflow -> runner: self-hosted"
echo
exec ./run.sh
