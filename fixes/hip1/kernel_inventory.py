#!/usr/bin/env python3
"""Turn the four plain-bf16 Tensile libraries into a launch plan.

Method 2 of the bf16 kernel audit (hip1.md §12.14): instead of asking hipBLASLt to pick a
kernel for a shape, hand every kernel in the library its own guarded operands and a shape
built from *its own* macro tile, so the question "does this kernel read past its operands"
is asked once per kernel rather than sampled through solution selection.

Inputs, both shipped with the stack and both read, never modified:

  <libdir>/TensileLibrary_..._Type_BB_..._gfx1201.co        the code object: `.kd` kernel
                                                            names + NT_AMDGPU_METADATA notes
                                                            (kernarg size, argument offsets,
                                                            workgroup size)
  <libdir>/TensileLibrary_..._Type_BB_..._gfx1201.dat.zlib  zlib + msgpack solution library:
                                                            every kernel's sizeMapping (macro
                                                            tile, depthU, workGroup, globalSplitU,
                                                            staggerU*, workGroupMapping*), which
                                                            is what the four packed scalars in
                                                            the argument block are built from
                                                            (hip1.md §12.9)

Outputs (written next to this script under resources/ unless --out-dir):

  bf16-kernel-inventory.json   one record per kernel: library, kernarg layout, tuning, the four
                               packed scalars, and either a launch plan or the reason it is
                               skipped (globalSplitU > 1, streamK, persistent, batched, ...)
  bf16-overrun-jobs.tsv        one line per (kernel, shape): everything launch_kernels.hip needs
                               to fill the argument block, launch, and check the result

The reference D is exact, so the launch check is a byte comparison rather than a tolerance: A and
B hold small integers, every partial sum is an integer below 241 (exactly representable in bf16),
and the accumulation order therefore cannot matter. Both sides compute the same closed form:

    A[i][l] = (2i + l) % 3              for l < 120, else 0
    B[l][j] = 1 if (2l + j) % 3 == 0 else 0
    D[i][j] = sum over l of A[i][l] * B[l][j] = P[(j % 3)][i]

with P[r][i] the partial sum over the l that are congruent to r modulo 3 — computed here in
O(M*K + M*N) and hashed as the raw bf16 bytes the kernel must write.

Usage:

    python3 fixes/hip1/kernel_inventory.py                 # all four libraries
    python3 fixes/hip1/kernel_inventory.py --check         # the fix3 kernel only, field by field
    python3 fixes/hip1/kernel_inventory.py --library Alik_Bljk --limit 20
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import struct
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXTRACTOR = HERE / "extract_kernel_codeobject.py"
DEFAULT_OUT_DIR = HERE / "resources"

TARGET = "hip-amdgcn-amd-amdhsa--gfx1201"
PLAIN_BF16 = re.compile(r"_Type_BB_")          # excludes Type_B8B8/Type_B8B/Type_B8F8/... too
NT_AMDGPU_METADATA = 32

# The fix3 kernel (hip1.md §12.9). --check holds the inventory against the numbers that were
# read out of this kernel's disassembly and its metadata by hand. The name is taken from
# launch_fault_kernel.hip rather than copied, so the two programs cannot drift apart.
FIX3_PROGRAM = HERE / "launch_fault_kernel.hip"
FIX3_EXPECT = {
    "kernarg": 144,
    "n_args": 30,
    "workgroup": 128,
    "group_seg": 1638,
    "macro_tile": [64, 128, 1],
    "depth_u": 16,
    "global_split_u": 1,
    "stagger_u": 32,
    "stagger_u_mapping": 1,
    "stagger_stride_shift": 3,
    "work_group_mapping": 8,
    "work_group_mapping_xcc": 1,
    "work_group_mapping_xcc_group": -1,
    "info0": 0x23200001,
    "info1": 0xFFC10008,
    "gemm_info": 0x00000001,
}


# --------------------------------------------------------------------------- minimal msgpack
#
# The stack's own libraries are msgpack, but only the system python here has the msgpack module
# (the training env does not), and this tool has to run in both. Types not used by these two
# files are still decoded, so a surprise in a future stack fails loudly instead of silently.


class MsgpackError(Exception):
    pass


def _mp_read(buf: bytes, pos: int):
    if pos >= len(buf):
        raise MsgpackError("truncated")
    b = buf[pos]
    pos += 1

    if b <= 0x7F:
        return b, pos
    if b >= 0xE0:
        return b - 0x100, pos
    if 0x80 <= b <= 0x8F:                                   # fixmap
        return _mp_map(buf, pos, b & 0x0F)
    if 0x90 <= b <= 0x9F:                                   # fixarray
        return _mp_array(buf, pos, b & 0x0F)
    if 0xA0 <= b <= 0xBF:                                   # fixstr
        n = b & 0x1F
        return _mp_str(buf, pos, n)
    if b == 0xC0:
        return None, pos
    if b == 0xC2:
        return False, pos
    if b == 0xC3:
        return True, pos
    if b == 0xC4:
        n, pos = _mp_uint(buf, pos, 1)
        return buf[pos:pos + n], pos + n                     # bin -> bytes
    if b == 0xC5:
        n, pos = _mp_uint(buf, pos, 2)
        return buf[pos:pos + n], pos + n
    if b == 0xC6:
        n, pos = _mp_uint(buf, pos, 4)
        return buf[pos:pos + n], pos + n
    if b == 0xCA:
        return struct.unpack_from(">f", buf, pos)[0], pos + 4
    if b == 0xCB:
        return struct.unpack_from(">d", buf, pos)[0], pos + 8
    if b in (0xCC, 0xCD, 0xCE, 0xCF):
        return _mp_uint(buf, pos, 1 << (b - 0xCC))
    if b in (0xD0, 0xD1, 0xD2, 0xD3):
        return _mp_int(buf, pos, 1 << (b - 0xD0))
    if b in (0xD9, 0xDA, 0xDB):
        n, pos = _mp_uint(buf, pos, 1 << (b - 0xD9))
        return _mp_str(buf, pos, n)
    if b in (0xDC, 0xDD):
        n, pos = _mp_uint(buf, pos, 2 << (b - 0xDC))
        return _mp_array(buf, pos, n)
    if b in (0xDE, 0xDF):
        n, pos = _mp_uint(buf, pos, 2 << (b - 0xDE))
        return _mp_map(buf, pos, n)
    if b in (0xC7, 0xC8, 0xC9):                              # ext -> (code, payload)
        n, pos = _mp_uint(buf, pos, 1 << (b - 0xC7))
        code = buf[pos]
        return (code, buf[pos + 1:pos + 1 + n]), pos + 1 + n
    raise MsgpackError(f"unsupported msgpack type 0x{b:02x} at {pos - 1}")


def _mp_uint(buf: bytes, pos: int, size: int):
    return int.from_bytes(buf[pos:pos + size], "big"), pos + size


def _mp_int(buf: bytes, pos: int, size: int):
    return int.from_bytes(buf[pos:pos + size], "big", signed=True), pos + size


def _mp_str(buf: bytes, pos: int, n: int):
    return buf[pos:pos + n].decode("utf-8", "replace"), pos + n


def _mp_array(buf: bytes, pos: int, n: int):
    out = []
    for _ in range(n):
        item, pos = _mp_read(buf, pos)
        out.append(item)
    return out, pos


def _mp_map(buf: bytes, pos: int, n: int):
    out = {}
    for _ in range(n):
        key, pos = _mp_read(buf, pos)
        value, pos = _mp_read(buf, pos)
        out[key] = value
    return out, pos


def msgpack_loads(buf: bytes):
    value, pos = _mp_read(buf, 0)
    if pos != len(buf):
        raise MsgpackError(f"{len(buf) - pos} trailing bytes")
    return value


# --------------------------------------------------------------------------- library discovery


def find_libdir(explicit: str | None) -> Path:
    candidates = [explicit, os.environ.get("AXL_HIPBLASLT_LIBDIR")]
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        for site in sorted(Path(prefix).glob("lib/python*/site-packages/_rocm_sdk_libraries/lib")):
            candidates.append(str(site / "hipblaslt/library/gfx1201"))
    candidates += [
        "/opt/rocm/core/lib/hipblaslt/library/gfx1201",
        "/opt/rocm/lib/hipblaslt/library/gfx1201",
    ]
    for cand in candidates:
        if cand and Path(cand).is_dir():
            return Path(cand)
    sys.exit("no hipBLASLt gfx1201 library directory found — pass --libdir or set "
             "AXL_HIPBLASLT_LIBDIR")


def library_layout(name: str) -> str:
    m = re.search(r"_l_([A-Za-z]+_[A-Za-z]+)_Cijk_", name)
    return m.group(1) if m else "?"


def bf16_libraries(libdir: Path, only: str | None) -> list[tuple[str, Path, Path]]:
    """(library file name, code object, .dat) for every plain-bf16 library."""
    out = []
    for co in sorted(libdir.glob("*.co")):
        if not PLAIN_BF16.search(co.name):
            continue
        dat = co.parent / (co.name[: -len(".co")] + ".dat.zlib")
        if not dat.is_file():
            print(f"[warn] {co.name}: no {dat.name} next to it, skipped", file=sys.stderr)
            continue
        if only and only not in co.name:
            continue
        out.append((co.name, co, dat))
    return out


def load_extractor():
    spec = importlib.util.spec_from_file_location("extract_kernel_codeobject", EXTRACTOR)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- code object


def code_object_kernels(ex, blob: bytes) -> dict[str, dict]:
    """Every `.kd` kernel in the code object, with its metadata note resolved."""
    sections = ex.elf_sections(blob)
    names = {s["name"][: -len(".kd")] for s in ex.elf_symbols(blob, sections)
             if s["name"].endswith(".kd")}

    notes: dict[str, dict] = {}
    for sec in sections:
        if sec["type"] != 7:                                # SHT_NOTE
            continue
        off, end = sec["offset"], sec["offset"] + sec["size"]
        while off < end:
            namesz, descsz, ntype = struct.unpack_from("<III", blob, off)
            off += 12
            off += (namesz + 3) // 4 * 4
            desc = blob[off:off + descsz]
            off += (descsz + 3) // 4 * 4
            if ntype != NT_AMDGPU_METADATA:
                continue
            try:
                md = msgpack_loads(desc)
            except MsgpackError as exc:
                print(f"[warn] unreadable metadata note: {exc}", file=sys.stderr)
                continue
            for kernel in (md or {}).get("amdhsa.kernels", []):
                notes[kernel[".name"]] = kernel

    out: dict[str, dict] = {}
    for name in sorted(names):
        md = notes.get(name)
        if md is None:
            out[name] = {"metadata": None}
            continue
        args = [{"name": a[".name"], "offset": a[".offset"], "size": a[".size"],
                 "kind": a.get(".value_kind", "")}
                for a in md.get(".args", [])]
        out[name] = {
            "kernarg_segment_size": md[".kernarg_segment_size"],
            "group_segment_fixed_size": md[".group_segment_fixed_size"],
            "private_segment_fixed_size": md[".private_segment_fixed_size"],
            "max_flat_workgroup_size": md[".max_flat_workgroup_size"],
            "sgpr_count": md[".sgpr_count"],
            "vgpr_count": md[".vgpr_count"],
            "args": args,
        }
    return out


def dat_solutions(dat: Path) -> dict[str, dict]:
    lib = msgpack_loads(zlib.decompress(dat.read_bytes(), 15))
    out: dict[str, dict] = {}
    for sol in lib.get("solutions", []):
        name = sol.get("kernelName") or sol.get("name")
        if name and name not in out:                        # first solution wins
            out[name] = sol
    return out


# --------------------------------------------------------------------------- tuning -> scalars


def packed_scalars(sm: dict) -> dict:
    """The four scalars of the argument block, from the solution's own tuning (hip1.md §12.9)."""
    gsu = int(sm.get("globalSplitU") or 1)
    info0 = (
        (gsu & 0x3FFF)
        | ((1 if sm.get("globalSplitUWorkGroupMappingRoundRobin") else 0) << 14)
        | ((1 if sm.get("globalSplitUCoalesced") else 0) << 15)
        | ((int(sm.get("staggerU") or 0) & 0xFF) << 16)
        | ((int(sm.get("staggerStrideShift") or 0) & 0x1F) << 24)
        | ((int(sm.get("staggerUMapping") or 0) & 0x7) << 29)
    )
    wgm = int(sm.get("workGroupMapping") or 0)
    xcc = int(sm.get("workGroupMappingXCC") or 0)
    xccg = int(sm.get("workGroupMappingXCCGroup") or 0)
    info1 = ((wgm & 0xFFFF) | ((xcc & 0x3F) << 16) | ((xccg & 0x3FF) << 22))
    return {
        "gemm_info": 0x00000001,                            # arg type 0 (inline), 1 GEMM
        "info0": info0,
        "info1": info1,
        "global_split_u": gsu,
        "stagger_u": int(sm.get("staggerU") or 0),
        "stagger_u_mapping": int(sm.get("staggerUMapping") or 0),
        "stagger_stride_shift": int(sm.get("staggerStrideShift") or 0),
        "work_group_mapping": wgm,
        "work_group_mapping_xcc": xcc,
        "work_group_mapping_xcc_group": xccg,
    }


def skip_reason(sm: dict, meta: dict, args: list[dict]) -> str | None:
    """Launch semantics we have not established; skipped and counted, never guessed."""
    if meta is None:
        return "no metadata note"
    if int(sm.get("globalSplitU") or 1) != 1:
        return "globalSplitU > 1 (needs the split-U workspace)"
    if int(sm.get("streamK") or 0) != 0:
        return "streamK (persistent/stream-K grid semantics)"
    if int(sm.get("prefetchAcrossPersistent") or 0) != 0:
        return "persistent kernel"
    if int(sm.get("globalAccumulation") or 0) not in (0, 2):
        return f"globalAccumulation={sm.get('globalAccumulation')}"
    if int(sm.get("packBatchDims") or 0) != 0:
        return "packBatchDims (batched free dims)"
    if sm.get("debugKernel"):
        return "debugKernel"
    if sm.get("CustomKernelName"):
        return "CustomKernelName"
    for want in ("Gemm info", "kernel info0", "kernel info1", "numWG", "SizesFree0", "SizesSum0",
                 "D", "C", "A", "B", "strideD0", "strideA0", "strideB0", "alpha"):
        if want not in {a["name"] for a in args}:
            return f"argument block has no '{want}'"
    return None


def layout_strides(layout: str, m: int, n: int, k: int) -> dict:
    """(strideA0, strideA1, strideB0, strideB1) in elements, per the library's layout token.

    `Ailk_Bjlk` is the pair hip1.md §12.9 read out of the fix3 kernel: A row-major M×K, B
    column-major K×N. The other three follow the same naming, and every kernel's control shape
    (a problem that fills its tiles exactly, checked against the exact reference D) fails loudly
    if a layout's strides are wrong, so a bad guess cannot be mistaken for an overrun.
    """
    a0, a1 = (k, 1) if layout.startswith("Ailk") else (1, m)
    b0, b1 = (1, n) if layout.endswith("Bjlk") else (n, 1)
    return {"strideA0": a0, "strideA1": a1, "strideB0": b0, "strideB1": b1}


# --------------------------------------------------------------------------- shapes and reference

REF_LIMIT = 120                # A is zero beyond this l, so every D stays below 241


def shape_ladder(mt_m: int, mt_n: int, depth_u: int) -> list[dict]:
    """The shapes one kernel is asked about: one control that must be clean, then the edges.

    Every shape uses the kernel's *own* macro tile, so "partial" means partial for this kernel.
    """
    full_k = 4 * depth_u
    tail_k = full_k + 8 if (full_k + 8) % depth_u else full_k + 4
    min_k = depth_u + 8 if (depth_u + 8) % depth_u else depth_u + 4
    m_edge = max(1, mt_m - 8)
    n_edge = max(1, mt_n - 8)

    def shape(label: str, m: int, n: int, k: int) -> dict:
        return {
            "label": label,
            "M": m, "N": n, "K": k,
            # SizesFree0 is the kernel's M and SizesFree1 its N, and the library's own dispatch
            # shows grid.x = M / MT_M (fixes/fix3's live fault: 20 workgroups = 1280 / 64)
            "grid": [-(-m // mt_m), -(-n // mt_n), 1],
            # The slotted tier puts each row so that its last byte is the last mapped byte of its
            # own mapping, which leaves the row start at (-K*2) mod 4096: a multiple of 32 only
            # when K*2 is. Shapes that miss this stay in the packed tier.
            "slotted": (k * 2) % 32 == 0,
        }

    return [
        shape("control-full-tiles", 2 * mt_m, 2 * mt_n, full_k),
        shape("k-tail", 2 * mt_m, 2 * mt_n, tail_k),
        shape("m-edge+k-tail", m_edge, 2 * mt_n, tail_k),
        shape("n-edge+k-tail", 2 * mt_m, n_edge, tail_k),
        shape("m-edge", m_edge, 2 * mt_n, full_k),
        shape("n-edge", 2 * mt_m, n_edge, full_k),
        shape("m+n-edges+k-tail", m_edge, n_edge, tail_k),
        shape("min", m_edge, n_edge, min_k),
        shape("edges-minus-4+k-tail", max(1, mt_m - 4), max(1, mt_n - 4), tail_k),
        shape("multi-tile-edges", 2 * mt_m + 8, 2 * mt_n + 8, tail_k),
    ]


def fill_a(i: int, l: int) -> int:
    return (2 * i + l) % 3 if l < REF_LIMIT else 0


def fill_b(l: int, j: int) -> int:
    return 1 if (2 * l + j) % 3 == 0 else 0


def bf16_bytes(value: int) -> bytes:
    """The kernel's D bytes for one value: exact small integers, so bf16 = the high half of f32."""
    return struct.pack("<H", struct.unpack("<I", struct.pack("<f", float(value)))[0] >> 16)


def d_bytes(m: int, n: int, k: int) -> bytes:
    """The exact D the kernel must write, in the kernel's own D layout: column-major over (i, j).

    D's descriptor is (i, j, batch) with i contiguous (hip1.md §12.14), so element (i, j) sits at
    i + j * m and the buffer reads column by column.

    B[l][j] is 1 exactly when (2l + j) % 3 == 0, i.e. when l % 3 == j % 3, so
    D[i][j] = sum over the l of that residue class of (2i + l) % 3: three partial sums per row of
    A, and D[i][j] depends on j only through j % 3, so the buffer is n repetitions of one of three
    m-element columns built once. Every value stays below 241, exactly representable in bf16, so
    the accumulation order cannot matter and the comparison is on bytes.
    """
    limit = min(k, REF_LIMIT)
    sums = []
    for i in range(m):
        row = [0, 0, 0]
        for l in range(limit):
            row[l % 3] += fill_a(i, l)
        sums.append(row)
    columns = [b"".join(bf16_bytes(sums[i][r]) for i in range(m)) for r in range(3)]
    return b"".join(columns[j % 3] for j in range(n))


def d_crc(m: int, n: int, k: int) -> int:
    """zlib crc32 of those bytes — cheap here, and 15 lines to reproduce on the device side."""
    return zlib.crc32(d_bytes(m, n, k)) & 0xFFFFFFFF


# --------------------------------------------------------------------------- main


def build_record(lib_name: str, layout: str, kernel: str, meta: dict | None, sol: dict | None):
    if sol is None:
        return {"kernel": kernel, "library": lib_name, "layout": layout, "status": "no-solution",
                "reason": "the .dat has no solution for this kernel name"}
    sm = sol.get("sizeMapping") or {}
    args = (meta or {}).get("args", [])
    reason = skip_reason(sm, meta, args)
    tiles = sm.get("macroTile") or [0, 0, 0]
    depth_u = int(sm.get("depthU") or 0)
    if reason is None and (not tiles[0] or not tiles[1] or not depth_u):
        reason = "macroTile/depthU missing from the solution"
    scalars = packed_scalars(sm)
    record = {
        "kernel": kernel,
        "library": lib_name,
        "layout": layout,
        "status": "skip" if reason else "launch",
        "reason": reason,
        "kernarg_segment_size": (meta or {}).get("kernarg_segment_size"),
        "workgroup": (meta or {}).get("max_flat_workgroup_size"),
        "group_segment_fixed_size": (meta or {}).get("group_segment_fixed_size"),
        "sgpr": (meta or {}).get("sgpr_count"),
        "vgpr": (meta or {}).get("vgpr_count"),
        "args": args,
        "macro_tile": [int(v) for v in tiles],
        "matrix_instruction": sm.get("matrixInstruction"),
        "depth_u": depth_u,
        "work_group": sm.get("workGroup"),
        **scalars,
    }
    if reason is None:
        record["shapes"] = shape_ladder(int(tiles[0]), int(tiles[1]), depth_u)
    return record


def write_jobs(path: Path, records: list[dict]) -> int:
    header = ["library", "kernel", "layout", "kernarg", "args", "info0", "info1", "workgroup",
              "grid", "M", "N", "K", "slotted", "label", "dcrc", "group_seg"]
    jobs = 0
    with path.open("w") as fh:
        fh.write("\t".join(header) + "\n")
        for rec in records:
            if rec["status"] != "launch":
                continue
            arg_map = ";".join(f"{a['name']}={a['offset']}" for a in rec["args"])
            for shape in rec["shapes"]:
                fh.write("\t".join(str(v) for v in [
                    rec["library"], rec["kernel"], rec["layout"], rec["kernarg_segment_size"],
                    arg_map, f"0x{rec['info0']:08x}", f"0x{rec['info1']:08x}", rec["workgroup"],
                    ",".join(str(g) for g in shape["grid"]), shape["M"], shape["N"], shape["K"],
                    int(shape["slotted"]), shape["label"],
                    f"0x{d_crc(shape['M'], shape['N'], shape['K']):08x}",
                    rec["group_segment_fixed_size"],
                ]) + "\n")
                jobs += 1
    return jobs


def fix3_kernel_name() -> str:
    """The kernel name launch_fault_kernel.hip launches, out of its own KERNEL literal."""
    text = FIX3_PROGRAM.read_text()
    body = text.split("static const char* KERNEL =", 1)[1].split(";", 1)[0]
    return "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', body))


def check_fix3(records: list[dict]) -> int:
    want_name = fix3_kernel_name()
    hits = [r for r in records if r["kernel"] == want_name]
    if len(hits) != 1:
        print(f"CHECK FAIL: {len(hits)} kernels match the name in launch_fault_kernel.hip, "
              f"expected exactly 1 ({want_name[:60]}…, {len(want_name)} chars)")
        return 1
    rec = hits[0]
    print(f"kernel            : {rec['kernel'][:72]}…  ({len(rec['kernel'])} chars)")
    print(f"library           : {rec['library']}")
    ok = True
    flat = {
        "kernarg": rec["kernarg_segment_size"],
        "n_args": len(rec["args"]),
        "workgroup": rec["workgroup"],
        "group_seg": rec["group_segment_fixed_size"],
        "macro_tile": rec["macro_tile"],
        "depth_u": rec["depth_u"],
        "stagger_u": rec["stagger_u"],
        "stagger_u_mapping": rec["stagger_u_mapping"],
        "stagger_stride_shift": rec["stagger_stride_shift"],
        "global_split_u": rec["global_split_u"],
        "work_group_mapping": rec["work_group_mapping"],
        "work_group_mapping_xcc": rec["work_group_mapping_xcc"],
        "work_group_mapping_xcc_group": rec["work_group_mapping_xcc_group"],
        "info0": rec["info0"],
        "info1": rec["info1"],
        "gemm_info": rec["gemm_info"],
    }
    for key, want in FIX3_EXPECT.items():
        got = flat.get(key)
        mark = "ok  " if got == want else "FAIL"
        if got != want:
            ok = False
        print(f"  {mark} {key:30} {got!r:22} (hip1.md §12.9: {want!r})")
    print(f"  shapes: " + ", ".join(f"{s['label']} {s['M']}x{s['N']}x{s['K']}"
                                   f"{'' if s['slotted'] else ' (packed only)'}"
                                   for s in rec["shapes"]))
    print("CHECK " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--libdir", default=None, help="hipBLASLt gfx1201 library directory")
    ap.add_argument("--library", default=None,
                    help="only libraries whose file name contains this (e.g. Alik_Bjlk)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N kernels (per library)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--check", action="store_true",
                    help="print the fix3 kernel's decoded rows and compare them with hip1.md §12.9")
    ap.add_argument("--quiet", action="store_true", help="no per-library summary")
    ap.add_argument("--ref-crc", nargs=3, type=int, metavar=("M", "N", "K"),
                    help="print the expected D crc for one shape and exit (for --shape runs)")
    ap.add_argument("--ref-fill", default="formula", choices=("formula", "ones"),
                    help="with --ref-crc: the operand fill the run uses")
    args = ap.parse_args()

    if args.ref_crc:
        m, n, k = args.ref_crc
        if args.ref_fill == "ones":
            # every product is 1, so D is the number of terms the kernel accumulated: K when it
            # reads exactly the declared operands
            blob = bf16_bytes(k) * (m * n)
            print(f"0x{zlib.crc32(blob) & 0xffffffff:08x}\t# M={m} N={n} K={k}, fill ones -> D == {k}")
            return 0
        print(f"0x{d_crc(m, n, k):08x}\t# M={m} N={n} K={k}, "
              f"{(k * 2) % 32 == 0 and 'slotted-eligible' or 'packed only'}")
        return 0

    libdir = find_libdir(args.libdir)
    print(f"libdir            : {libdir}")
    ex = load_extractor()
    records: list[dict] = []
    for lib_name, co, dat in bf16_libraries(libdir, args.library):
        layout = library_layout(lib_name)
        _, payload = ex.read_ccob(str(co))
        blob = ex.take_target(payload, TARGET)
        kernels = code_object_kernels(ex, blob)
        solutions = dat_solutions(dat)
        count = 0
        for kernel in sorted(kernels):
            records.append(build_record(lib_name, layout, kernel, kernels[kernel],
                                        solutions.get(kernel)))
            count += 1
            if args.limit and count >= args.limit:
                break
        if not args.quiet:
            launch = sum(1 for r in records if r["library"] == lib_name and r["status"] == "launch")
            skipped = sum(1 for r in records if r["library"] == lib_name and r["status"] == "skip")
            print(f"{layout:11} {lib_name[:52]:52} kernels {count:4d}  "
                  f"launchable {launch:4d}  skipped {skipped:4d}")

    if args.check:
        return check_fix3(records)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv = out_dir / "bf16-kernel-inventory.json"
    inv.write_text(json.dumps(
        {"libdir": str(libdir), "target": TARGET, "kernels": records}, indent=1) + "\n")
    jobs = out_dir / "bf16-overrun-jobs.tsv"
    n_jobs = write_jobs(jobs, records)

    launch = sum(1 for r in records if r["status"] == "launch")
    skipped = [r for r in records if r["status"] == "skip"]
    print(f"\nkernels           : {len(records)} ({launch} launchable, {len(skipped)} skipped)")
    if skipped:
        reasons: dict[str, int] = {}
        for rec in skipped:
            reasons[rec["reason"]] = reasons.get(rec["reason"], 0) + 1
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  skip x{n:<5} {reason}")
    print(f"shapes per kernel : {next((len(r['shapes']) for r in records if r['status'] == 'launch'), 0)}")
    print(f"jobs              : {n_jobs}  -> {jobs}")
    print(f"inventory         : {inv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
