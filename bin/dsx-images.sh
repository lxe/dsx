#!/usr/bin/env bash
# dsx image switcher for the mcmp10 host (2x RTX 4090, CUDA 13.2, sm_89).
#
# It selects one of the locally built images, stops/removes the live model container, and starts the selected image
# with the same serving configuration that the current live container uses. It never builds implicitly while switching
# and never touches images outside the three dsx tags below.
#
#   original        dsx:t16-sm89-avx2-maddubs
#                   baseline before the prompt-boundary checkpoint patch (built 2026-10-26)
#   prompt-prefix   dsx:t16-sm89-avx2-maddubs-prefix-checkpoints
#                   prompt-boundary checkpoint fix (commit 3982ca0, built 2026-10-30)
#   branch          dsx:t16-sm89-avx2-maddubs-session-cache
#                   current fix/prefix-checkpoints sources with the bounded multi-session KV/prefix cache
#                   (built from this working tree; aliases: session, current, prefix-checkpoints)
#
# The switch is destructive: it unloads the running 252 GB model before starting the new container. A snapshot of the
# old container inspect/logs is written before removal. Use --dry-run to see the exact commands, or run without
# --yes and answer the interactive prompt.
#
# Examples:
#   ./bin/dsx-images.sh status
#   ./bin/dsx-images.sh original --dry-run
#   ./bin/dsx-images.sh prompt-prefix --yes
#   ./bin/dsx-images.sh branch --yes --wait
#   ./bin/dsx-images.sh build branch           # rebuild current sources -> branch image
#   ./bin/dsx-images.sh stop --yes             # unload only, no new container
#
# Serving defaults match the live container: ctx 524288, arena 90 GB, max-tokens 56000, host port 8096 -> 8090,
# model dir /var/lib/llama-cpp/models/DeepSeek-V4.1-Flash-1017/dsx-t16, GPUs 0,1, DSX_CPU_ISA=avx2.
#
# Environment overrides (all optional):
#   NAME, MODEL_DIR, CTX, ARENA_GB, MAX_TOKENS, HOST_BIND, HOST_PORT, CONTAINER_PORT,
#   DSX_GPUS, DSX_CPU_ISA, STOP_TIMEOUT, READY_TIMEOUT,
#   CACHE_SESSIONS (default 3), SESSION_CACHE_GB (default 6), PREFIX_ANCHOR_TOKENS (default 16384),
#   DSX_STATE_DIR (default ~/.cache/dsx-images), API_KEY.
#
# The cache flags are passed only for the branch image (and explicit images whose tag contains "session-cache");
# original/prefix binaries do not know those flags.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

NAME="${NAME:-deepseek-v41-dsx-t16-avx2}"
MODEL_DIR="${MODEL_DIR:-/var/lib/llama-cpp/models/DeepSeek-V4.1-Flash-1017/dsx-t16}"
MODEL_FILE="${MODEL_FILE:-/models/DeepSeek-V4.1-Flash-1017-dsx-t16.gguf}"
RANK_FILE="${RANK_FILE:-/models/expert-rank.json}"
MODEL_NAME="${MODEL_NAME:-deepseek-v4.1-flash}"
CTX="${CTX:-524288}"
ARENA_GB="${ARENA_GB:-90}"
MAX_TOKENS="${MAX_TOKENS:-56000}"
HOST_BIND="${HOST_BIND:-172.25.255.123}"
HOST_PORT="${HOST_PORT:-8096}"
CONTAINER_PORT="${CONTAINER_PORT:-8090}"
DSX_GPUS="${DSX_GPUS:-0,1}"
DSX_CPU_ISA="${DSX_CPU_ISA:-avx2}"
STOP_TIMEOUT="${STOP_TIMEOUT:-180}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"
CACHE_SESSIONS="${CACHE_SESSIONS:-3}"
SESSION_CACHE_GB="${SESSION_CACHE_GB:-6}"
PREFIX_ANCHOR_TOKENS="${PREFIX_ANCHOR_TOKENS:-16384}"
DSX_STATE_DIR="${DSX_STATE_DIR:-$HOME/.cache/dsx-images}"
API_KEY="${API_KEY:-}"

IMAGE_ORIGINAL="${IMAGE_ORIGINAL:-dsx:t16-sm89-avx2-maddubs}"
IMAGE_PREFIX="${IMAGE_PREFIX:-dsx:t16-sm89-avx2-maddubs-prefix-checkpoints}"
IMAGE_SESSION="${IMAGE_SESSION:-dsx:t16-sm89-avx2-maddubs-session-cache}"

TARGET=""
ACTION="switch"
DRY_RUN=0
ASSUME_YES=0
WAIT=0

die() { printf 'error: %s\n' "$*" >&2; exit 2; }

usage() {
    sed -n '2,38p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<'USAGE_EOF'

options:
  original | prompt-prefix | branch   image to run
                                      aliases: base/baseline -> original;
                                               prefix, prefix-patch -> prompt-prefix;
                                               session, session-cache, current, prefix-checkpoints -> branch
  build [branch]              build the branch image from the current sources (refuses original/prompt-prefix tags)
  status | list                show the live container and the three image tags; touches nothing
  stop                         stop/remove the live container only (no new container)
  --image IMG                  explicit image tag instead of one of the three aliases
  --dry-run                    print the commands; do not stop, remove, build or run anything
  --yes                        required for a real switch; without it an interactive prompt is used
  --wait                       after starting, poll /health until READY_TIMEOUT seconds
  --cache-sessions N           branch image only (default 3)
  --session-cache-gb GB        branch image only (default 6; host memory is tight here)
  --prefix-anchor-tokens N     session image only (default 16384; 0 disables anchors)
  --stop-timeout SEC           docker stop timeout (default 180)
  --ready-timeout SEC          --wait timeout (default 1800)
  --help                       this help
USAGE_EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        original|base|baseline) TARGET=original; shift ;;
        prefix|prompt-prefix|prefix-patch|prefix-fix) TARGET=prefix; shift ;;
        session|session-cache|current|fix|branch|prefix-checkpoints|prefix-checkpoints-v2) TARGET=session; shift ;;
        build) ACTION=build; shift; [[ $# -gt 0 && "$1" != -* ]] && { TARGET="$1"; shift; } ;;
        status|list) ACTION=status; shift ;;
        stop) ACTION=stop; shift ;;
        --image) [[ $# -gt 1 ]] || die "--image needs a tag"; TARGET="image:$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --yes|-y) ASSUME_YES=1; shift ;;
        --wait) WAIT=1; shift ;;
        --cache-sessions) [[ $# -gt 1 ]] || die "--cache-sessions needs a number"; CACHE_SESSIONS="$2"; shift 2 ;;
        --session-cache-gb) [[ $# -gt 1 ]] || die "--session-cache-gb needs GB"; SESSION_CACHE_GB="$2"; shift 2 ;;
        --prefix-anchor-tokens) [[ $# -gt 1 ]] || die "--prefix-anchor-tokens needs a number"; PREFIX_ANCHOR_TOKENS="$2"; shift 2 ;;
        --stop-timeout) [[ $# -gt 1 ]] || die "--stop-timeout needs seconds"; STOP_TIMEOUT="$2"; shift 2 ;;
        --ready-timeout) [[ $# -gt 1 ]] || die "--ready-timeout needs seconds"; READY_TIMEOUT="$2"; shift 2 ;;
        --name) [[ $# -gt 1 ]] || die "--name needs a value"; NAME="$2"; shift 2 ;;
        --host-port) [[ $# -gt 1 ]] || die "--host-port needs a value"; HOST_PORT="$2"; shift 2 ;;
        --ctx) [[ $# -gt 1 ]] || die "--ctx needs a value"; CTX="$2"; shift 2 ;;
        --arena-gb) [[ $# -gt 1 ]] || die "--arena-gb needs a value"; ARENA_GB="$2"; shift 2 ;;
        --max-tokens) [[ $# -gt 1 ]] || die "--max-tokens needs a value"; MAX_TOKENS="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
done

have_docker() { command -v docker >/dev/null || die "docker not found"; }
container_exists() { docker inspect "$NAME" >/dev/null 2>&1; }
container_running() { docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null | grep -qx true; }
image_exists() { docker image inspect "$1" >/dev/null 2>&1; }

image_for_target() {
    case "$TARGET" in
        original) printf '%s' "$IMAGE_ORIGINAL" ;;
        prefix) printf '%s' "$IMAGE_PREFIX" ;;
        session) printf '%s' "$IMAGE_SESSION" ;;
        image:*) printf '%s' "${TARGET#image:}" ;;
        *) die "choose one of: original | prompt-prefix | branch (or pass --image TAG)" ;;
    esac
}

show_status() {
    have_docker
    echo "container: $NAME"
    if container_exists; then
        docker inspect "$NAME" --format '  image={{.Config.Image}}  state={{.State.Status}}  id={{.Id}}  started={{.State.StartedAt}}'
        docker ps --filter "name=^/${NAME}$" --format '  running  ports={{.Ports}}'
    else
        echo "  not present"
    fi
    echo "images:"
    for pair in "original:$IMAGE_ORIGINAL" "prompt-prefix:$IMAGE_PREFIX" "branch:$IMAGE_SESSION"; do
        label="${pair%%:*}"; tag="${pair#*:}"
        if image_exists "$tag"; then
            info="$(docker image inspect "$tag" --format '{{.Id}}  created={{.Created}}  size={{.Size}}')"
            printf '  %-8s %s  %s\n' "$label" "$tag" "$info"
        else
            printf '  %-8s %s  MISSING\n' "$label" "$tag"
        fi
    done
    if [[ -f "$DSX_STATE_DIR/current-image" ]]; then
        echo "last switched image: $(cat "$DSX_STATE_DIR/current-image")"
    fi
}

snapshot_live() {
    local dir="$DSX_STATE_DIR/snapshots/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$dir"
    if container_exists; then
        docker inspect "$NAME" > "$dir/container-inspect.json" 2>/dev/null || true
        docker logs --tail 2000 "$NAME" > "$dir/container.log" 2>&1 || true
        PREVIOUS_IMAGE="$(docker inspect "$NAME" --format '{{.Config.Image}}' 2>/dev/null || true)"
        printf '%s\n' "$PREVIOUS_IMAGE" > "$DSX_STATE_DIR/previous-image"
    fi
    echo "snapshot: $dir"
}

stop_live() {
    if container_exists; then
        echo "stopping $NAME (timeout ${STOP_TIMEOUT}s)"
        docker stop -t "$STOP_TIMEOUT" "$NAME" >/dev/null 2>&1 || true
        docker rm -f "$NAME" >/dev/null 2>&1 || true
        echo "container removed"
    else
        echo "no live container named $NAME"
    fi
}

confirm_or_dry_run() {
    local image="$1"
    local prev="${2:-}"
    if [[ "$DRY_RUN" == 1 ]]; then
        echo "DRY RUN: would switch"
        printf '  previous: %s\n  new:      %s\n' "${prev:-none}" "$image"
        printf '  docker stop -t %q %q\n' "$STOP_TIMEOUT" "$NAME"
        printf '  docker rm -f %q\n' "$NAME"
        printf '  docker run -d --name %q --gpus all --ipc=host --security-opt label=disable --shm-size 64m --ulimit memlock=-1:-1 --restart no ' "$NAME"
        printf -- '-e DSX_GPUS=%q -e DSX_CPU_ISA=%q -p %q -v %q %q ' "$DSX_GPUS" "$DSX_CPU_ISA" "${HOST_BIND}:${HOST_PORT}:${CONTAINER_PORT}" "${MODEL_DIR}:/models:ro" "$image"
        printf 'dsx-server --model %q --rank %q --ctx %q --arena-gb %q --max-tokens %q --host 0.0.0.0 --port %q --model-name %q' \
            "$MODEL_FILE" "$RANK_FILE" "$CTX" "$ARENA_GB" "$MAX_TOKENS" "$CONTAINER_PORT" "$MODEL_NAME"
        if [[ "$TARGET" == session || "$image" == *session-cache* ]]; then
            printf ' --cache-sessions %q --session-cache-gb %q --prefix-anchor-tokens %q' "$CACHE_SESSIONS" "$SESSION_CACHE_GB" "$PREFIX_ANCHOR_TOKENS"
        fi
        if [[ -n "$API_KEY" ]]; then
            printf ' --api-key %q' "***hidden***"
        fi
        echo
        return 0
    fi
    if [[ "$ASSUME_YES" != 1 ]]; then
        if [[ -t 0 ]]; then
            local answer
            read -r -p "Unload ${prev:-the live container} and start $image? [y/N] " answer
            [[ "$answer" =~ ^[Yy]$ ]] || die "cancelled"
        else
            die "refusing non-interactive switch; pass --yes or --dry-run"
        fi
    fi
}

wait_health() {
    local url="http://${HOST_BIND}:${HOST_PORT}/health"
    local deadline=$(( $(date +%s) + READY_TIMEOUT ))
    echo "waiting for $url (up to ${READY_TIMEOUT}s)"
    while (( $(date +%s) < deadline )); do
        if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
            echo "server is healthy: $url"
            return 0
        fi
        if ! container_running; then
            echo "container exited; last logs:" >&2
            docker logs --tail 80 "$NAME" >&2 || true
            return 1
        fi
        sleep 3
    done
    echo "readiness timeout after ${READY_TIMEOUT}s; container may still be loading" >&2
    return 1
}

do_build() {
    have_docker
    [[ "$TARGET" == "session" || "$TARGET" == "" ]] || die "build is only allowed for session (refusing to overwrite $TARGET)"
    [[ -f "$ROOT/Dockerfile.mcmp10" ]] || die "Dockerfile.mcmp10 not found under $ROOT"
    echo "building $IMAGE_SESSION (branch image) from $ROOT (current working tree)"
    docker build -f "$ROOT/Dockerfile.mcmp10" -t "$IMAGE_SESSION" "$ROOT"
    echo "built $IMAGE_SESSION"
}

do_stop() {
    have_docker
    confirm_or_dry_run "${NAME} (no new container)" ""
    [[ "$DRY_RUN" == 1 ]] && { printf '  docker stop -t %q %q\n  docker rm -f %q\n' "$STOP_TIMEOUT" "$NAME" "$NAME"; return 0; }
    snapshot_live
    stop_live
}

do_switch() {
    have_docker
    local image prev
    image="$(image_for_target)"
    image_exists "$image" || {
        echo "image '$image' is not present locally." >&2
        echo "build it with: $0 build branch" >&2
        exit 1
    }
    [[ -d "$MODEL_DIR" ]] || die "model dir not found: $MODEL_DIR"
    [[ -f "$MODEL_DIR/$(basename "$MODEL_FILE")" ]] || die "model file not found under $MODEL_DIR"
    [[ -f "$MODEL_DIR/$(basename "$RANK_FILE")" ]] || die "rank file not found under $MODEL_DIR"

    prev="$(docker inspect "$NAME" --format '{{.Config.Image}}' 2>/dev/null || true)"
    if [[ -z "$prev" && -f "$DSX_STATE_DIR/current-image" ]]; then
        prev="$(cat "$DSX_STATE_DIR/current-image")"
    fi

    confirm_or_dry_run "$image" "${prev:-none}"
    [[ "$DRY_RUN" == 1 ]] && return 0

    mkdir -p "$DSX_STATE_DIR"
    snapshot_live
    stop_live

    local args=(
        run -d --name "$NAME"
        --gpus all
        --ipc=host
        --security-opt label=disable
        --shm-size 64m
        --ulimit memlock=-1:-1
        --restart no
        -e "DSX_GPUS=$DSX_GPUS"
        -e "DSX_CPU_ISA=$DSX_CPU_ISA"
        -p "${HOST_BIND}:${HOST_PORT}:${CONTAINER_PORT}"
        -v "${MODEL_DIR}:/models:ro"
        "$image"
        dsx-server
        --model "$MODEL_FILE"
        --rank "$RANK_FILE"
        --ctx "$CTX"
        --arena-gb "$ARENA_GB"
        --max-tokens "$MAX_TOKENS"
        --host 0.0.0.0
        --port "$CONTAINER_PORT"
        --model-name "$MODEL_NAME"
    )
    # only the multi-session image accepts these flags; the older images abort on unknown arguments
    if [[ "$TARGET" == session || "$image" == *session-cache* ]]; then
        args+=(--cache-sessions "$CACHE_SESSIONS" --session-cache-gb "$SESSION_CACHE_GB" --prefix-anchor-tokens "$PREFIX_ANCHOR_TOKENS")
    fi
    [[ -n "$API_KEY" ]] && args+=(--api-key "$API_KEY")

    local cid
    cid="$(docker "${args[@]}")" || { echo "docker run failed" >&2; exit 1; }
    printf '%s\n' "$image" > "$DSX_STATE_DIR/current-image"
    printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${prev:-none}" "$image" >> "$DSX_STATE_DIR/history.log"
    echo "started $cid as $NAME ($image)"
    echo "listen http://${HOST_BIND}:${HOST_PORT}"
    echo "logs: docker logs -f $NAME"
    if [[ "$WAIT" == 1 ]]; then
        wait_health || exit 1
    fi
}

case "$ACTION" in
    status) show_status ;;
    build) do_build ;;
    stop) do_stop ;;
    switch) do_switch ;;
    *) die "unknown action" ;;
esac
