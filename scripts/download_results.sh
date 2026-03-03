#!/bin/bash

set -euo pipefail

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"
results_path="$repo_path/results"
cache_dir="${DOWNLOAD_CACHE_DIR:-/tmp/vcno-rbno-results-cache}"

all_problems=(
    "elasticity"
    "poisson_setup1"
    "poisson_setup2"
)

usage() {
    cat <<'EOF'
Usage:
  bash scripts/download_results.sh <base_url> [problem ...]

Examples:
  bash scripts/download_results.sh https://example.com/vcno-rbno-results
  bash scripts/download_results.sh https://example.com/vcno-rbno-results elasticity poisson_setup1

The base URL must contain the files produced by scripts/package_results_for_download.sh,
including manifest.tsv and all split archive parts.
EOF
}

download_file() {
    local url="$1"
    local destination="$2"

    if command -v curl >/dev/null 2>&1; then
        curl -L --fail --retry 3 --output "$destination" "$url"
        return
    fi

    if command -v wget >/dev/null 2>&1; then
        wget -O "$destination" "$url"
        return
    fi

    printf 'Either curl or wget is required.\n' >&2
    exit 1
}

contains_problem() {
    local candidate="$1"
    shift

    for problem in "$@"; do
        if [[ "$problem" == "$candidate" ]]; then
            return 0
        fi
    done

    return 1
}

if [[ $# -lt 1 ]]; then
    usage >&2
    exit 1
fi

base_url="${1%/}"
shift

if [[ $# -eq 0 ]]; then
    selected_problems=("${all_problems[@]}")
else
    selected_problems=("$@")
fi

for selected_problem in "${selected_problems[@]}"; do
    if ! contains_problem "$selected_problem" "${all_problems[@]}"; then
        printf 'Unknown problem: %s\n' "$selected_problem" >&2
        usage >&2
        exit 1
    fi
done

mkdir -p "$cache_dir"
mkdir -p "$results_path"

bash "$repo_path/scripts/create_results_directories.sh" >/dev/null

manifest_path="$cache_dir/manifest.tsv"
download_file "$base_url/manifest.tsv" "$manifest_path"

declare -A archive_parts
declare -A archive_problem

while IFS=$'\t' read -r problem result_unit part_name checksum _size_bytes; do
    if [[ -z "$problem" ]]; then
        continue
    fi

    if ! contains_problem "$problem" "${selected_problems[@]}"; then
        continue
    fi

    part_path="$cache_dir/$part_name"

    download_file "$base_url/$part_name" "$part_path"

    actual_checksum="$(sha256sum "$part_path" | awk '{print $1}')"
    if [[ "$actual_checksum" != "$checksum" ]]; then
        printf 'Checksum mismatch for %s\n' "$part_name" >&2
        exit 1
    fi

    archive_key="${problem}:${result_unit}"
    archive_problem["$archive_key"]="$problem"
    archive_parts["$archive_key"]+="$part_path"$'\n'
done < "$manifest_path"

for archive_key in "${!archive_parts[@]}"; do
    archive_problem_name="${archive_problem[$archive_key]}"

    if ! contains_problem "$archive_problem_name" "${selected_problems[@]}"; then
        continue
    fi

    mapfile -t part_paths < <(printf '%s' "${archive_parts[$archive_key]}" | sed '/^$/d' | sort)

    if [[ "${#part_paths[@]}" -eq 0 ]]; then
        continue
    fi

    cat "${part_paths[@]}" | tar -xzf - -C "$results_path"
done

printf 'Downloaded and extracted results into %s\n' "$results_path"
