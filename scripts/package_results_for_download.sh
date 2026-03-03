#!/bin/bash

set -euo pipefail

repo_path="$(cd "$(dirname "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)")" && pwd)"
results_path="$repo_path/results"
output_dir="${1:-/tmp/vcno-rbno-results-download}"
part_size="${PART_SIZE:-1900m}"

problems=(
    "elasticity"
    "poisson_setup1"
    "poisson_setup2"
)

result_units=(
    "train_dataset"
    "test_dataset"
    "model_train_outputs"
    "model_test_outputs"
)

manifest_path="$output_dir/manifest.tsv"

mkdir -p "$output_dir"
: > "$manifest_path"

for problem in "${problems[@]}"; do
    for result_unit in "${result_units[@]}"; do
        source_path="$results_path/$problem/$result_unit"

        if [[ ! -d "$source_path" ]]; then
            printf 'Missing directory: %s\n' "$source_path" >&2
            exit 1
        fi

        archive_prefix="results-${problem}-${result_unit}.tar.gz.part-"
        rm -f "$output_dir/$archive_prefix"*

        tar -czf - -C "$results_path" "$problem/$result_unit" \
            | split -d -a 3 -b "$part_size" - "$output_dir/$archive_prefix"

        for part_path in "$output_dir/$archive_prefix"*; do
            checksum="$(sha256sum "$part_path" | awk '{print $1}')"
            size_bytes="$(stat -c%s "$part_path")"

            printf '%s\t%s\t%s\t%s\t%s\n' \
                "$problem" \
                "$result_unit" \
                "$(basename "$part_path")" \
                "$checksum" \
                "$size_bytes" \
                >> "$manifest_path"
        done
    done
done

printf 'Wrote split archives and manifest to %s\n' "$output_dir"
