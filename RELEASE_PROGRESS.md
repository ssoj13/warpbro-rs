# Release and NVIDIA PR checkpoint — 2026-10-13

## Latest checkpoint (supersedes the historical preparation state below)

All four NVIDIA PRs are submitted as normal OPEN/non-draft PRs, from our fork:
[1509 constant memory](https://github.com/NVIDIA/cuda-rust/pull/1509),
[1510 Cargo profiles](https://github.com/NVIDIA/cuda-rust/pull/1510),
[1511 inline intent](https://github.com/NVIDIA/cuda-rust/pull/1511), and
[1512 scalar floors](https://github.com/NVIDIA/cuda-rust/pull/1512).
No direct upstream push or merge was attempted. Source branches are
nvidia/fix/constant-memory-lto (5e34bad84), nvidia/fix/release-profile-pins (f6aeeceda),
nvidia/feat/inline-intent (9f8cee663), nvidia/fix/scalar-floors (4d5f4d2ad).
Original Alex Khal author/DCO is preserved on the first three; current sign-off is added.
Current pinned-nightly Linux tests, backend tests, clippy, fmt and SPDX pass.
GPU examples pass: constant_memory with device LTO and two 256-element rewrites;
const_generic VALUE=4/8; addressof_sharedarray pointer/device-call and raw-receiver checks.
Scalar generator/codegen tests, ABI/current-output and book stamp checks pass;
the canonical Linux scalar catalog/book stamp is 12fe4e6c, with only target/backend
fields of 64 records changed. Other 965 records/all 1029 ABI identities stay identical.
The final scalar checks are in nvidia-scalar-{linux-verified,final-guards}.log.
Every branch's SPDX/fmt receipt is nvidia-<task>-guards.log. Read current PR CI/reviews
before any further upstream action; reviewers may need to authorize hosted runs.

Our compiler main/origin is 758c069c0f02fa099d8feb42ea56a70865c3c452.
It contains backport 6dc3dc7a00ad196fc0caaf4e6cdab00e40fcab85 and standalone backend-pin commit
5e15109343fadd82d73ec95dc8a737ebdbfebb60 (the CLI pins backport 6dc3dc7). The final commit only corrects the CI guard:
retained measurement provenance and its generated reference may name the actual dated
LLVM nightly; installation instructions still reject a dated nightly. Both the positive
and injected-negative checks passed, then hosted status-guard 38462813650 passed all
14 jobs. No production compiler source changed after 5e1510934. Original untracked
crates/ and inst.cmd were preserved. Own tests: 318 generator tests, 1221 codegen units,
261 cargo-oxide units plus setup integration, clippy and fmt pass; ABI ledger remains
1029. Private native backend release build passes. The real Rust scalar project compiles
--arch sm_75 and runs on RTX 3080 Ti with 256 bit-exact mul/fma results.
Receipts: fork-rust-probe-{build-verified,run}.log, compiler-ci-failure.log,
compiler-guard-push.log; local positive/negative guard helper fork_parity_check.sh.
Fixture/backend remain on D:/codex-checks-20261013; global installed tools are untouched.

WarpBro's release candidate changes exactly 13 CUDA Git lock pins and bootstrap.py
from a11d4cc to the tested 5e1510934; Cargo.toml/windows-sys and package version 0.1.0
are unchanged. CHANGELOG.md describes the reissue. Local locked full CUDA suite:
315 passed, 0 failed, 13 ignored (56.72s runtime). Release build with embedded cubins
sm_75/sm_86/sm_89/sm_120 passed in 10m39s. Clippy --release --locked --tests -D warnings
passed in 2m57s; fmt passed. Actual release executable --warmup-cuda reports RTX 3080 Ti
ready in 200.5955ms. Receipts: reissue-full-tests-retry.log, reissue-cubin-build.log,
reissue-clippy.log, reissue-fmt.log and reissue-warmup.log. Isolated target is
D:/codex-checks-20261013/warp-target, using the private verified backend and pinned LLVM.

Original release archives are retained under D:/codex-checks-20261013/warp-release-before;
API SHA-256/size and ZIP CRC passed (warp-release-backup-verified.json). Original
annotated tag object ad335f72a2ec973b9ef774b160b0de08fa331681, peeled source
86def178c188f34f26978d7be12e66013c532ffb, is preserved on GitHub as
archive/v0.1.0-before-20261013. The published v0.1.0 remains intact until tag CI succeeds.
Next: commit/push the verified candidate, move v0.1.0 only as authorized, wait both tag
builds and publishing, then verify the replacement archive digests, CRC and contents.

Environmental precautions: Windows Git checkout in NVIDIA trees had CRLF in imported.json
and Bash scripts. Linux hash/script checks failed honestly. Restore canonical Git-index
bytes for these checks; do not change upstream.lock hashes to hide the mismatch. Linux
Git cannot read Windows absolute .git worktree files, so scripts export GIT_DIR and
GIT_WORK_TREE explicitly. The completed NVIDIA worktrees were cleaned afterward; scalar
intrinsic files may display as dirty solely for LF/CRLF (inspect actual diff before restore).
NVIDIA-index main stamping differing from old raw-Windows stamp was resolved by regeneration
and independent semantic/count checks. Source scripts were not weakened.

Heavy build outputs must stay on D:; C: recently had about 2 GB free. Use the global
Windows buildlock around WSL and native commands. Background shader/compiler jobs from
other sessions are not ours and must not be killed or cleaned.

The current task continues the verified recovery in HANDOFF.md. The operator asked
to merge completed work into our own main branches and reissue WarpBro v0.1.0 on
GitHub. Never merge or push directly into NVIDIA or ansidium. The latest operator
instruction authorizes submitting NVIDIA PRs **after all applicable checks pass**.
An earlier temporary hold on sending PRs is superseded by that conditional authorization.

## Published state

- WarpBro main/origin: 60bbb75 before this documentation checkpoint. The MIS source
  integration is 9f7ef5b; Windows/Linux CI 38430261140 succeeded. Fractal main/origin
  c4222f4 contains explicit a.mul_add(a, b * b). The strict material oracle is enabled.
- ansidium/cuda-rust-windows PR 14 is OPEN against ansidium/main, head 4891e74ea.
  Current CI: 56 SUCCESS, no failures or pending checks; 3 intentional skips and
  1 neutral aggregate. No account push permission to ansidium. Do not attempt merge.
- NVIDIA/cuda-rust accepts contributions. Maintainer nihalpasham explicitly invited
  this scalar-floor fix on issue 1462 on October 10. Its current main is bc46b1ffb.
- There are four source PRs by ssoj13 in ansidium: 10 constant memory, 11 inline intent,
  12 Cargo profile pins (all merged there), and 14 scalar arithmetic floors (open).
  NVIDIA has no existing PR by ssoj13. No new NVIDIA PR was sent in this follow-up.
- Our compiler fork main/origin remains a11d4cce8. Original checkout
  C:/projects/projects.rust.cg/nv/cuda-oxide-windows has pre-existing untracked crates/
  and inst.cmd: preserve them. Installed cargo-oxide/backend have not been changed.
- WarpBro v0.1.0 remains unchanged: annotated tag ad335f72a2ec973b9ef774b160b0de08fa331681
  points to 86def178c188f34f26978d7be12e66013c532ffb. Do not bump the package version.

## Local compiler integration

Owned worktree: C:/Users/joss1/AppData/Local/Temp/codex-mis-20261013/cuda-main,
branch integrate/scalar-fp-main, parent origin/main a11d4cce8. Changes are staged,
not committed. It backports the scalar policy/tests/evidence from PR 14 and
regenerates its own outputs; it does not merge the diverged ansidium branch.

The fork pins LLVM 23.1.0 / 16696adcd119e6ba9cc175207d984d7021211acb. PR 14 pins
LLVM 23.1.3 / aab11ee0. Reusing the newer profile correctly fails the validator.
The old pinned LLVM was downloaded into target/mis-fix/tools-20260828 (no toolchain
change); all 192 old-version codegen/assembly probes and 64 canonicalizations pass.
The new old-version evidence profile is rust-llvm-23.1.0-16696adc-scalar-baseline.
Unused new-version evidence was removed from this worktree only.

Completed: 316 generator units + 2 architecture integration tests; 1221 codegen units;
doctests pass. Retried clippy with CARGO_INCREMENTAL=0 passes after a disk-full error.
Generator check, ABI-history check, and final fmt are queued in session 14621.
Logs: target/mis-fix/fork-{tests,clippy-retry,check,abi,fmt}.log.
Semantic audit: only target/backend fields on 64 scalar records differ; the other
965 records, all 1029 ABI identities, signatures and opcodes are identical.
Diff audit: 1312 files change catalog-hash comments only. Catalog/book stamp 40054757.
Ruff check/format pass for both probe helpers. The LLVM helper now reads upstream.lock
with tomllib instead of a hardcoded revision fallback.

GitNexus repo name cuda-scalar-main indexes this checkout. Impact for the two existing
policy functions is LOW (1 direct caller and 0 direct callers). detect_changes(all)
fails on the unchanged Windows directory symlink mathdx_ffi_test/tools, the already
reported BUG3.md defect; staged fallback passes, 78 symbols/1330 files, CRITICAL due
to regeneration volume. Warn before commit and retain the independent semantic/diff
audit. Reindex after any further edits and commit. Use detect_changes before commits.

## NVIDIA transplants (all local, based on bc46b1ffb)

All worktrees below share the prefix C:/Users/joss1/AppData/Local/Temp/codex-mis-20261013/.
They have staged changes and no new commits yet. Indexes are separately registered.

| Worktree / branch | Source | Current verification |
| --- | --- | --- |
| nvidia-constant / nvidia/fix/constant-memory-lto | 041e65c5a | Linux 56+99 exporter tests and 3+304+147 lowering tests pass, clippy passes. Final fmt needs explicit confirmation because the first helper script was edited while running and its final line failed. |
| nvidia-profile / nvidia/fix/release-profile-pins | d61153e19 | Linux cargo-oxide 246 tests, clippy and fmt pass, session 56225 exit 0. |
| nvidia-inline / nvidia/feat/inline-intent | af61bb486 | Dependency tests/clippy running, then rustc-codegen-cuda backend tests/clippy. Active session 69387. |
| nvidia-scalar / nvidia/fix/scalar-floors | d39feb6eb + 4891e74ea | Queued session 31983; runs generate, tests, current/ABI checks, clippy, book stamp and fmt. |

Inline transplant conflicts were reviewed and resolved: preserve NVIDIA's new
reference-validity text/behavior and replace the obsolete alwaysinline-only query
with the complete InlineAttr mapping. Remaining code auto-merges; the new reference
facts stay intact. Fixed one stale rustdoc link INLINE_INTENT_ATTR to InlineIntent.
Restage that file before commit. It may still have CHERRY_PICK_HEAD from resolving
the --no-commit cherry-pick: inspect git status before committing.

Both global/function export impacts are CRITICAL: 2 direct callers, 7 execution flows.
This was reported to the operator. Test both IR dialects and GPU examples before
submission. The other affected production-function impacts are LOW.

NVIDIA's unchanged tests use /dev/null and Unix path comparisons: initial native
Windows runs fail 6 exporter cases and 2 cargo-oxide cases. Preserve these logs;
do not weaken assertions. Linux reruns exercise the actual tests successfully.
NVIDIA pins nightly-2026-10-11, now installed in WSL Ubuntu with rustc-dev/src,
LLVM tools, clippy and fmt. CUDA 13.3 is /usr/local/cuda. All builds use the global
Windows buildlock wrapper around wsl, not a separate Linux lock.

Active script: target/mis-fix/nvidia_checks_d.sh; Linux target directory is
/mnt/d/codex-checks-20261013/nvidia-linux-target. Do not edit a script while its bash
process is running. Earlier scripts using bash -lc unquoted PATH failed on parentheses;
the file-based invocation fixes this. The scalar worktree book was temporarily
stamped using raw CRLF JSON (15757a51); restore its staged 3601e99b book version before
the queued script reaches book checks, or recalculate the stamp after generate.
Scalar helper tomllib change is unstaged and needs restaging.

## Storage and receipts

All receipts/helpers are under WarpBro target/mis-fix/ (ignored). A Windows compiler
clippy initially failed with disk error 112. C: recently had about 5 GB free, D: about
176 GB. New heavy build outputs should use D:/codex-checks-20261013/; avoid creating
another large target on C:. The owned completed nvidia-profile/cuda-oxide/target was
moved, with absolute-path checks, to D:/codex-checks-20261013/nvidia-profile-target-before.
No shared/original compiler cache or unrelated work was deleted.

Old WarpBro release metadata: target/mis-fix/warp-release-before.json. Both old ZIPs
are downloaded to D:/codex-checks-20261013/warp-release-before; verify SHA-256 and CRC
against metadata before relying on that backup. Draft replacement notes are in
target/mis-fix/warp-release-notes.md. Release rules are cached in ignored
.omh/RELEASE_RULE.md. Existing workflow .github/workflows/ci.yml builds two OS archives
on v* tags, embeds sm_75/86/89/120 cubins plus PTX, and publishes after both builds pass.

## Finish in order

1. Poll active checks and fix actual failures. Verify generated-book stamps using LF
   catalog bytes, not a CRLF checkout. Run final fmt/SPDX checks and focused GPU examples;
   consider a real sm_75 scalar Rust kernel to verify the original compiler gate end to end.
2. Prepare four concise NVIDIA PR bodies linking the source ansidium PRs and actual current
   receipts. Review diffs and DCO, detect changes, commit on each fork branch, reindex.
   Confirm remote is ssoj13/cuda-rust-windows before pushing; create PRs against NVIDIA/main.
   Do not directly push/merge into NVIDIA or ansidium. Latest user condition permits sending
   only after applicable verification passes.
3. Commit the verified compiler-main backport and fast-forward our fork main to it without
   touching its pre-existing untracked files. Push our main. Keep evidence on LLVM 23.1.0;
   do not float toolchain/catalog pins or merge unrelated ansidium changes.
4. Finish WarpBro release notes/changelog/current handoff. If adopting the new compiler,
   update bootstrap.py's compiler Git pin and the three CUDA Git lock sources by exact SHA;
   never cargo update. Verify with the intended backend, strict GPU oracle, release/cubin
   build and smoke. Use isolated tool/backend artifacts to avoid disturbing other sessions.
5. Preserve the old release/tag, create a backup tag, then move annotated v0.1.0 to the
   verified main candidate and force-update only that authorized tag. Wait for tag CI,
   verify both new archives (SHA-256, CRC, layout and Windows runtime), and publish the
   prepared notes. Keep old assets available until replacement artifacts are verified.
6. Record the real final SHAs, PR URLs, CI and release evidence; report anything still
   pending. Do not call a prepared branch merged or a release regenerated before it is.
