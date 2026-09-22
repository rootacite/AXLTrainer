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
| `amdfq-pocs-2026-09.tar.zst.gpg` | 952 985 B | `b7dc40fcc73fe78cae563b8c1e5098ca97909f3089df67f12eea0e8dc4635c6a` | The original C tail guard (`amdfq/amdfq-tail`, sources and CMake builds), the HIP VMM churn/integrity probe (`amdfq/vmm-cc`), the two cross-process arms (`amdfq/vmm-ru`, `amdfq/vmm-ru.oneshot-wip`), the Vulkan/RADV arm (`amdfq/fk-vmm`), the hipMalloc-stream replay tool and its captured streams (`amdfq/replay`, `amdfq/streams`), and the cost-comparison notes (`amdfq/eva-2.md`). |
| `gfx1201-disclosure-2026-09.tar.zst.gpg` | 3 296 783 B | `5a588b75aa3830976f913e909995159d80386cf74ffcaa1a04fdf7354877f130` | The PoC binaries and sources with their evidence tree (`amdfq/final`: `exploit1*`, `half-raw-2s*`, `evi/`, `tools/`), and the vendor security-report package (`amdfq/doc/psirt`: report, email, attachments, evidence, poc, pictures). |
| `gfx1201-research-2026-09.tar.zst.gpg` | 27 313 131 B | `919e647a324c2a61c8770a96b602c0bddde97dd06df3c4332f93ea2290c0b4e6` | The writeups (`amdfq/doc`: route comparisons, live measurement notes, the overread narrative, the VA-never-reuse pair, the torch-test ladder and its stdout extracts), the kernel-overread conclusions (`conclusions/`), the field reports and repros (`fixes/`: fix1, fix2, fix3, hip1), and the HIP probes (`tools/hip/`). |

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

The same files also exist, unencrypted, at `/storage/amdfq_sealed/` (`/storage` is the NFS mount on
the author's network), with the original relative paths preserved — `amdfq/vmm-ru`, `conclusions/`,
`fixes/`, `tools/hip` … 951 files, 125 MB. It needs no passphrase, and it is the recovery path if the
key file is ever lost. It is a mirror, not a working copy: change the bundles and this together.

## Residue: git history

The removal was committed as a working-tree change only. Everything that was tracked before it —
`amdfq/streams`, `amdfq/replay`, most of `amdfq/doc`, `conclusions/`, `fixes/`, `HIP.md`'s old
references, `tools/hip/` — is still reachable in this repository's history
(`git log --diff-filter=D`, `git show <commit>:<path>`). Rewriting that history has not been done;
until it is, the working tree is not the only place this material lives.

## How the bundles were made

```bash
# per bundle, from the repo root; <paths> are the original relative paths
tar -C <repo> -cf - <paths> | zstd -19 -T0 \
  | gpg --batch --yes --symmetric --cipher-algo AES256 \
        --passphrase-file ~/.axl-archive-key -o archive/<name>.tar.zst.gpg

# and the plaintext mirror, paths preserved
cd <repo> && rsync -aR --checksum <paths> /storage/amdfq_sealed/
```

Before the working tree copies were deleted, both routes were verified against them: every bundle was
decrypted and `diff -r`-compared to its source paths, and the mirror was compared file-for-file
(951 files, no differences).
