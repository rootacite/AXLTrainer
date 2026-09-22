# archive — sealed research material

涉及负责任披露流程，暂不公开.

The material in this directory is the record of an AMD driver investigation and the PoCs, evidence,
field reports and writeups that came out of it. It is kept in the repository, encrypted, instead of
being published: the mechanism and the exploitation detail are part of an ongoing responsible
disclosure process (AMD PSIRT). Anything outside `archive/` refers to it as "sealed" rather than
linking to it.

Each bundle is `tar` → `zstd` → **gpg symmetric** (`AES256`). The tree inside a bundle keeps its
original paths, so unpacking restores exactly the layout the files had.

| Bundle | Size | sha256 | What is in it |
| --- | --- | --- | --- |
| `amdfq-pocs-2026-09.tar.zst.gpg` | 953 543 B | `8f28ed445ceb14eb230c3901de1cd25981f11ff84252ab850943691b86d772e2` | The original C tail guard (`amdfq/amdfq-tail`, sources and CMake builds), the HIP VMM churn/integrity probe (`amdfq/vmm-cc`), the two cross-process arms (`amdfq/vmm-ru`, `amdfq/vmm-ru.oneshot-wip`), the Vulkan/RADV arm (`amdfq/fk-vmm`), the hipMalloc-stream replay tool and its captured streams (`amdfq/replay`, `amdfq/streams`), and the cost-comparison notes (`amdfq/eva-2.md`). |
| `gfx1201-disclosure-2026-09.tar.zst.gpg` | 3 290 218 B | `1b4d1ab42a49986e76aafeb5343e74bea5e171d780b243e9dce488f66b98e0d3` | The PoC binaries and sources with their evidence tree (`amdfq/final`: `exploit1*`, `half-raw-2s*`, `evi/`, `tools/`), and the vendor security-report package (`amdfq/doc/psirt`: report, email, attachments, evidence, poc, pictures). |
| `gfx1201-research-2026-09.tar.zst.gpg` | 27 312 186 B | `2dbce26dd9bd676ae68fff31d9fbf42606133edf78b1934b1c6373989fa387f1` | The writeups (`amdfq/doc`: route comparisons, live measurement notes, the overread narrative, the VA-never-reuse pair, the torch-test ladder and its stdout extracts), the kernel-overread conclusions (`conclusions/`), the field reports and repros (`fixes/`: fix1, fix2, fix3, hip1), the HIP probes (`tools/hip/`), and this machine's HIP toolchain notes (`HIP.md`). |

Total: 31 MB in the working tree.

## Opening a bundle

The passphrase lives outside the repository, in `~/.axl-archive-key` (mode 600). It is 32 random bytes,
base64, generated when the bundles were made; **losing it makes the bundles unreadable**, which is why
the plaintext mirror below exists.

```bash
# list
gpg --batch --quiet --decrypt --passphrase-file ~/.axl-archive-key archive/<bundle>.tar.zst.gpg \
  | zstd -d | tar -t

# unpack somewhere else
mkdir -p /tmp/sealed && gpg --batch --quiet --decrypt --passphrase-file ~/.axl-archive-key \
    archive/<bundle>.tar.zst.gpg | zstd -d | tar -x -C /tmp/sealed
```

## Plaintext mirror

The same files also exist, unencrypted, at `/storage/amdfq_sealed/` (the NFS mount on the author's
network), with the original relative paths preserved — `amdfq/vmm-ru`, `conclusions/`, `fixes/`,
`tools/hip`, `HIP.md` … 952 files. It needs no passphrase, and it is the recovery path if the key file
is ever lost. It is a mirror, not a working copy: change the bundles and this together.

## Residue: git history

The removal was committed as a working-tree change only. Everything that was tracked before it —
`amdfq/streams`, `amdfq/replay`, most of `amdfq/doc`, `conclusions/`, `fixes/`, `HIP.md` — is still
reachable in this repository's history (`git log --diff-filter=D`, `git show <commit>:<path>`).
Rewriting that history has not been done; until it is, the working tree is not the only place this
material lives.

## Provenance

The first sealing run (2026-09-23 03:08) built and verified all three bundles against the working tree,
and the tree copies were deleted only after that: every bundle was decrypted and `diff -r`-compared to
its source paths, and the mirror was compared file-for-file (951 files, no differences).

A later run rebuilt the bundles and was interrupted by an unclean machine shutdown: the disclosure
bundle came back as a 0-byte file (its data never reached the disk) and the research bundle was left
one revision old. All three were then rebuilt from the mirror — which is why the mirror, not the
working tree, is the reference copy — and re-verified the same way (952 files, no differences, including
`HIP.md`). The passphrase and the mirror were untouched by that shutdown.

## How the bundles were made

```bash
# per bundle; <paths> are the original relative paths, and the mirror is the tar root because the
# working-tree copies are gone
tar -C /storage/amdfq_sealed -cf - <paths> | zstd -19 -T2 \
  | gpg --batch --yes --symmetric --cipher-algo AES256 \
        --passphrase-file ~/.axl-archive-key -o archive/<name>.tar.zst.gpg

# the plaintext mirror itself, paths preserved (run before the tree copies were deleted)
cd <repo> && rsync -aR --checksum <paths> /storage/amdfq_sealed/
```
