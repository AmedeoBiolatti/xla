#!/usr/bin/env bash
# Copyright 2026 The OpenXLA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -euo pipefail
export XLA_RESULT_DIR="$RUNNER_TEMP/xla-52711-results"
mkdir -p "$XLA_RESULT_DIR"
printf '%s  %s\n' "$XLA_SOURCE_SHA256" xla/pjrt/cpu/cpu_client.cc \
  "$XLA_TEST_SHA256" xla/pjrt/cpu/cpu_pjrt_compiler_test.cc | sha256sum --check
git fetch --no-tags --depth=1 origin "$XLA_BASE_SHA"
git diff "$XLA_BASE_SHA" -- xla/pjrt/cpu/cpu_client.cc \
  xla/pjrt/cpu/cpu_pjrt_compiler_test.cc > "$XLA_RESULT_DIR/final.patch"
clang-21 --version > "$XLA_RESULT_DIR/compiler.txt"
df -h > "$XLA_RESULT_DIR/disk-before.txt"
python3 - <<'PY'
import shutil
assert shutil.disk_usage('.').free >= 10_000_000_000, 'Need 10 GB free for cold source build'
PY

xla_source=xla/pjrt/cpu/cpu_client.cc
cp "$xla_source" "$XLA_RESULT_DIR/cpu_client.fixed.cc"
trap 'cp "$XLA_RESULT_DIR/cpu_client.fixed.cc" "$xla_source"' EXIT
git show "$XLA_BASE_SHA:$xla_source" > "$xla_source"
git diff --exit-code "$XLA_BASE_SHA" -- "$xla_source"

xla_bazel_options=(
  --batch --output_user_root="$RUNNER_TEMP/xla-bazel-root"
)
xla_test_options=(
  test //xla/pjrt/cpu:cpu_pjrt_compiler_test
  --test_output=errors --jobs=4 --local_resources=memory=6144
  --repository_cache="$RUNNER_TEMP/xla-repository-cache"
  --noannounce_rc --show_progress_rate_limit=15
  --http_timeout_scaling=0.2 --experimental_repository_downloader_retries=1
  --config=clang_local --repo_env=CC=/usr/bin/clang-21
  --repo_env=CXX=/usr/bin/clang++-21
  --@rules_ml_toolchain//common:stdlib=libstdc++
  --@rules_ml_toolchain//common:static_libcxx=false
)
xla_testlogs=bazel-testlogs/xla/pjrt/cpu/cpu_pjrt_compiler_test
set +e
"$XLA_BAZEL" "${xla_bazel_options[@]}" "${xla_test_options[@]}" \
  --test_filter=CpuPjrtCompilerTest.CompileMlirOpSuccess \
  2>&1 | tee "$XLA_RESULT_DIR/baseline-build.log"
xla_baseline_exit=${PIPESTATUS[0]}
set -e
test "$xla_baseline_exit" -eq 3
cp "$xla_testlogs/test.log" "$XLA_RESULT_DIR/baseline-test.log"
cp "$xla_testlogs/test.xml" "$XLA_RESULT_DIR/baseline-test.xml"
python3 - <<'PY'
import os
from pathlib import Path
import xml.etree.ElementTree as ET
root = ET.parse(Path(os.environ['XLA_RESULT_DIR']) / 'baseline-test.xml').getroot()
cases = root.findall('.//testcase')
assert len(cases) == 1 and cases[0].get('name') == 'CompileMlirOpSuccess'
failures = cases[0].findall('failure')
assert len(failures) == 1, 'Expected only the missing input-dump assertion to fail'
message = failures[0].get('message', '') + (failures[0].text or '')
assert 'Value of: dumps' in message and "whose size 0 doesn't match" in message, message
print('Confirmed baseline failure: module.mlir is not dumped before the override is applied.')
PY

cp "$XLA_RESULT_DIR/cpu_client.fixed.cc" "$xla_source"
trap - EXIT
printf '%s  %s\n' "$XLA_SOURCE_SHA256" "$xla_source" \
  "$XLA_TEST_SHA256" xla/pjrt/cpu/cpu_pjrt_compiler_test.cc | sha256sum --check
"$XLA_BAZEL" "${xla_bazel_options[@]}" "${xla_test_options[@]}" \
  '--test_filter=*' 2>&1 | tee "$XLA_RESULT_DIR/patched-build.log"
cp "$xla_testlogs/test.log" "$XLA_RESULT_DIR/patched-test.log"
cp "$xla_testlogs/test.xml" "$XLA_RESULT_DIR/patched-test.xml"
git diff --check
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET
results = Path(os.environ['XLA_RESULT_DIR'])
root = ET.parse(results / 'patched-test.xml').getroot()
cases = root.findall('.//testcase')
assert len(cases) == 7, f'Expected the full suite of 7 tests, got {len(cases)}'
assert all(not c.findall('failure') and not c.findall('error') and not c.findall('skipped') for c in cases)
assert any(c.get('name') == 'CompileMlirOpSuccess' for c in cases)
summary = {
    'status': 'passed',
    'base': os.environ['XLA_BASE_SHA'],
    'validation_commit': os.environ['GITHUB_SHA'],
    'source_sha256': os.environ['XLA_SOURCE_SHA256'],
    'test_sha256': os.environ['XLA_TEST_SHA256'],
    'patch_sha256': hashlib.sha256((results / 'final.patch').read_bytes()).hexdigest(),
    'baseline': {'tests': 1, 'failures': 1, 'reason': 'missing module.mlir dump'},
    'patched': {'tests': len(cases), 'failures': 0},
    'run_url': f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}",
}
(results / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
    output.write('Final patch validation: baseline regression fails as expected; patched full CPU compiler suite passes all 7 tests.\n')
PY
