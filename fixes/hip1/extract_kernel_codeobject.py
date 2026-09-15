#!/usr/bin/env python3
"""Extract the hipModule-loadable code object that holds the fix3 faulting kernel.

`TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co`,
which sits next to this script, is one member of hipBLASLt's gfx1201 Tensile library as
shipped in AMD's `CCOB` container: a zstd-compressed clang offload bundle.  The runtime
decompresses it inside `hipModuleLoad`, so it is a container, not a loadable object.  This
script produces the object inside it:

  1. check the CCOB header, decompress the payload (zstd);
  2. take the `hipv4-amdgcn-amd-amdhsa--gfx1201` item out of the offload bundle
     (`clang-offload-bundler`; falls back to the ELF boundary when the tool is absent);
  3. verify the result is an AMDGPU ELF and that the faulting kernel's symbols are in it,
     printing its entry point and kernel-descriptor (`.kd`) address;
  4. with `--load-check`, build and run a 40-line HIP program that hands the object to
     `hipModuleLoadData` and looks the kernel up with `hipModuleGetFunction` — no kernel is
     launched, so this loads nothing onto the device queue.

  python fixes/hip1/extract_kernel_codeobject.py                 # extract + verify
  python fixes/hip1/extract_kernel_codeobject.py --list          # show the bundle items
  python fixes/hip1/extract_kernel_codeobject.py --load-check    # + runtime load test
  python fixes/hip1/extract_kernel_codeobject.py --out /path/to/object.elf
  python fixes/hip1/extract_kernel_codeobject.py --kernel SUBSTR # another kernel
  python fixes/hip1/extract_kernel_codeobject.py --co OTHER.co --kernel KERNEL_NAME
                                                                 # another library's kernel

Default output is `/tmp/axl-hip1/` (the extracted object is ~29 MB, so it is kept out of the
repository unless `--out` says otherwise).  Background, the container format and the evidence
that this kernel is the faulting one: `fixes/hip1/hip1.md` §12.
"""

from __future__ import annotations

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CO = os.path.join(
    HERE,
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bjlk_Cijk_Dijk_gfx1201.co",
)
DEFAULT_KERNEL = (
    "Cijk_Ailk_Bjlk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x128x16_MI16x16x1_SN_LDSB0_AFC1_AG0_AGGSUA0_"
    "AGNTAB0_AFEM1_AFEM1_ASEM1_CD1_1_CLR1_CLS0_CADS0_DTLA0_DTLB0_DTLM0_DTVA1_DTVB1_DTVMXSA0_DTVMXSB0_"
    "DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_HPLR0_ISA1201_ICIW0_IU1_K1_"
    "LDSTI0_LBSPPA0_LBSPPB0_LBSPPMXSA0_LBSPPMXSB0_LBSPPM0_LPA0_LPB0_LPMXSA0_LPMXSB0_LPM0_LRVW8_LWPMn1_"
    "MIAV1_MIWT1_8_MXLIBL_MXSFNS_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTE0_NTMXSA0_NTMXSB0_NTM0_NTWS0_"
    "NVn1_NVA0_NVB0_NVC0_NVD0_NVE0_NVMXSA0_NVMXSB0_NVM0_NVWS0_NEPBS0_NLCA1_NLCB8_ONLL0_PAP0_PGL0_PGR1_"
    "PLR0_PKA0_SGROB0_SIA3_SS0_SPO0_SRVW0_SSO0_SVW8_SK0_SKFTR0_SKFDPO0_SKXCCM0_SNLL0_SIP1_SGRO0_TDMI0_"
    "TDMIM0_TDMS0_TIN0_THn1_THA0_THB0_THC0_THD0_THE0_THMXSA0_THMXSB0_THM0_THWS0_TLDS0_TLDSM1_ULSGRO0_"
    "USL1_USLMX0_UIOFGRO0_UPLRP0_USFGROn1_USI0_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG64_2_1"
)  # the kernel fixes/fix3 aborts in; hip1.md §12.1

DEFAULT_OUT_DIR = "/tmp/axl-hip1"
DEFAULT_TARGET = "hipv4-amdgcn-amd-amdhsa--gfx1201"

CCOB_MAGIC = b"CCOB"
BUNDLE_MAGIC = b"__CLANG_OFFLOAD_BUNDLE__"
ELF_MAGIC = b"\x7fELF"
EM_AMDGPU = 224
SHT_SYMTAB, SHT_DYNSYM = 2, 11
STT_OBJECT, STT_FUNC = 1, 2

LOAD_CHECK_SRC = r"""
// Loads a code object with hipModuleLoadData and looks a kernel up by name.
// Nothing is launched: this only proves the runtime accepts the object.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <vector>

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <code-object> <kernel name>\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror("fopen"); return 2; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> image(n);
    if (fread(image.data(), 1, n, f) != (size_t)n) { fprintf(stderr, "short read\n"); return 2; }
    fclose(f);

    hipModule_t module = nullptr;
    hipError_t e = hipModuleLoadData(&module, image.data());
    if (e != hipSuccess) {
        printf("hipModuleLoadData : FAILED  %s\n", hipGetErrorString(e));
        return 1;
    }
    printf("hipModuleLoadData : ok      module=%p (%ld bytes)\n", (void *)module, n);

    hipFunction_t fn = nullptr;
    e = hipModuleGetFunction(&fn, module, argv[2]);
    printf("hipModuleGetFunction: %s  function=%p\n", e == hipSuccess ? "ok" : hipGetErrorString(e), (void *)fn);
    hipModuleUnload(module);
    return e == hipSuccess ? 0 : 1;
}
"""


def read_ccob(path: str) -> tuple[dict, bytes]:
    """Validate a CCOB container and return (header fields, decompressed payload)."""
    blob = open(path, "rb").read()
    if blob[:4] != CCOB_MAGIC:
        sys.exit(f"{path}: not a CCOB container (magic {blob[:4]!r})")
    header = {
        "version": struct.unpack_from("<H", blob, 4)[0],
        "flags": struct.unpack_from("<H", blob, 6)[0],
        "compressed_size": struct.unpack_from("<Q", blob, 8)[0],
        "original_size": struct.unpack_from("<Q", blob, 16)[0],
        "checksum": blob[24:32].hex(),
        "file_size": len(blob),
    }
    body = blob[32:]
    original = header["original_size"]
    try:
        import zstandard  # type: ignore
    except ImportError:
        with tempfile.NamedTemporaryFile(suffix=".zst") as tmp:
            tmp.write(body)
            tmp.flush()
            payload = subprocess.run(["zstd", "-d", "-c", tmp.name], check=True, capture_output=True).stdout
    else:
        payload = zstandard.ZstdDecompressor().decompress(body, max_output_size=original + 64)
    if len(payload) != original:
        sys.exit(f"{path}: decompressed {len(payload)} bytes, header says {original}")
    return header, payload


def bundle_items(payload: bytes) -> list[str]:
    """List the targets inside a clang offload bundle, via clang when it is available."""
    bundler = shutil.which("clang-offload-bundler")
    if not bundler:
        return []
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        tmp.write(payload)
        tmp.flush()
        out = subprocess.run([bundler, "--list", "--type=o", "--input", tmp.name],
                             capture_output=True, text=True)
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def take_target(payload: bytes, target: str) -> bytes:
    """Return the code object for `target` from the offload bundle."""
    if payload[:24] != BUNDLE_MAGIC:
        return payload  # already a bare code object
    bundler = shutil.which("clang-offload-bundler")
    if bundler:
        with tempfile.TemporaryDirectory() as td:
            src = os.path.join(td, "payload.bin")
            dst = os.path.join(td, "target.o")
            open(src, "wb").write(payload)
            res = subprocess.run(
                [bundler, "--unbundle", "--type=o", f"--targets={target}", "--input", src, "--output", dst],
                capture_output=True, text=True)
            if res.returncode == 0 and os.path.exists(dst):
                return open(dst, "rb").read()
            sys.exit(f"clang-offload-bundler --unbundle failed for {target}:\n{res.stderr.strip()}")
    # Fallback: this container holds one host stub (empty) and the device object last, so the
    # object runs from the ELF magic to the end of the payload.  Validated below.
    start = payload.find(ELF_MAGIC)
    if start < 0:
        sys.exit("no clang-offload-bundler and no ELF inside the payload")
    return payload[start:]


def elf_sections(blob: bytes) -> list[dict]:
    if blob[:4] != ELF_MAGIC or blob[4] != 2:
        sys.exit("not an ELF64 object")
    machine = struct.unpack_from("<H", blob, 0x12)[0]
    if machine != EM_AMDGPU:
        sys.exit(f"unexpected e_machine {machine} (AMDGPU is {EM_AMDGPU})")
    shoff = struct.unpack_from("<Q", blob, 0x28)[0]
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", blob, 0x3A)
    raw = []
    for i in range(shnum):
        off = shoff + i * shentsize
        sh_name, sh_type = struct.unpack_from("<II", blob, off)
        sh_addr, sh_offset, sh_size = struct.unpack_from("<QQQ", blob, off + 16)
        sh_link = struct.unpack_from("<I", blob, off + 40)[0]
        sh_entsize = struct.unpack_from("<Q", blob, off + 56)[0]
        raw.append({"name_index": sh_name, "type": sh_type, "addr": sh_addr, "offset": sh_offset,
                    "size": sh_size, "link": sh_link, "entsize": sh_entsize})
    strtab = raw[shstrndx]
    for ent in raw:
        ent["name"] = cstring(blob, strtab["offset"] + ent["name_index"])
    return raw


def cstring(blob: bytes, offset: int) -> str:
    end = blob.find(b"\0", offset)
    return blob[offset:end].decode("utf-8", "replace")


def elf_symbols(blob: bytes, sections: list[dict]) -> list[dict]:
    """Every entry of .symtab/.dynsym, with names resolved through the linked string table."""
    symbols = []
    for ent in sections:
        if ent["type"] not in (SHT_SYMTAB, SHT_DYNSYM) or not ent["entsize"]:
            continue
        strtab = sections[ent["link"]]
        for i in range(ent["size"] // ent["entsize"]):
            off = ent["offset"] + i * ent["entsize"]
            st_name, st_info, _st_other, st_shndx = struct.unpack_from("<IBBH", blob, off)
            st_value, st_size = struct.unpack_from("<QQ", blob, off + 8)
            if not st_name:
                continue
            symbols.append({
                "name": cstring(blob, strtab["offset"] + st_name),
                "type": st_info & 0xF,
                "section": st_shndx,
                "value": st_value,
                "size": st_size,
                "table": ent["name"],
            })
    return symbols


def report(blob: bytes, kernel: str) -> int:
    sections = elf_sections(blob)
    symbols = elf_symbols(blob, sections)
    descriptors = [s for s in symbols if s["name"].endswith(".kd") and kernel in s["name"]]
    kernels = {s["name"] for s in symbols if s["name"].endswith(".kd")}
    print(f"code object       : {len(blob)} bytes, {len(kernels)} kernels, "
          f"{len(sections)} sections, e_machine=AMDGPU")

    if not descriptors:
        print(f"kernel            : NOT FOUND (no .kd symbol contains {kernel!r})")
        return 1

    by_name = {s["name"]: s for s in descriptors}
    if len(by_name) > 1:
        print(f"kernel            : {len(by_name)} tunings match {kernel!r} — pass the full name "
              f"(hip1.md §12.1) to single one out:")
        for name in sorted(by_name):
            print(f"                    .kd 0x{by_name[name]['value']:x}  …{name[-28:]}")
        return 1

    name = next(iter(by_name))
    stem = name[: -len(".kd")]
    entry = next((s for s in symbols if s["name"] == stem and s["type"] == STT_FUNC), None)
    desc = by_name[name]
    print(f"kernel            : {len(stem)}-char name, exactly one match")
    if entry:
        print(f"entry point       : 0x{entry['value']:x}  ({entry['table']})")
    print(f"kernel descriptor : 0x{desc['value']:x}  ({desc['size']} B, section {desc['section']})"
          f"   <- 'kernel_obj' in the fault log")
    return 0


def find_hipcc() -> str | None:
    for cand in (shutil.which("hipcc"),
                 os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "bin", "hipcc"),
                 "/opt/rocm/bin/hipcc"):
        if cand and os.path.exists(cand):
            return cand
    return None


def load_check(obj_path: str, kernel: str) -> int:
    """hipModuleLoadData + hipModuleGetFunction on the extracted object; nothing is launched."""
    hipcc = find_hipcc()
    if not hipcc:
        print("load check        : skipped, no hipcc (see tools/hip/setup_rocm_dev.sh)")
        return 0
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "load_check.cpp")
        exe = os.path.join(td, "load_check")
        open(src, "w").write(LOAD_CHECK_SRC)
        build = subprocess.run([hipcc, "-O3", src, "-o", exe], capture_output=True, text=True)
        if build.returncode != 0:
            print("load check        : hipcc failed\n" + build.stdout + build.stderr)
            return 1
        res = subprocess.run([exe, obj_path, kernel], capture_output=True, text=True)
        print(res.stdout.rstrip())
        if res.returncode != 0:
            print(res.stderr.rstrip())
            return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Format, evidence and the fault this kernel is part of: fixes/hip1/hip1.md §12")
    ap.add_argument("--co", default=DEFAULT_CO, help="CCOB file to read (default: the one next to this script)")
    ap.add_argument("--out", default=None, help=f"where to write the code object (default: {DEFAULT_OUT_DIR}/)")
    ap.add_argument("--target", default=DEFAULT_TARGET, help="offload-bundle target to extract")
    ap.add_argument("--kernel", default=DEFAULT_KERNEL, help="kernel-name substring to look for")
    ap.add_argument("--list", action="store_true", help="list the bundle's targets and stop")
    ap.add_argument("--load-check", action="store_true", help="also load it via hipModuleLoadData/GetFunction")
    args = ap.parse_args()

    header, payload = read_ccob(args.co)
    print(f"container         : CCOB v{header['version']}, flags {header['flags']}, "
          f"{header['file_size']} B compressed -> {len(payload)} B payload")
    print(f"payload           : {payload[:24].decode('ascii', 'replace')}")

    if args.list:
        for item in bundle_items(payload):
            print(f"  bundle item     : {item}")
        return 0

    blob = take_target(payload, args.target)
    out = args.out or os.path.join(DEFAULT_OUT_DIR, os.path.basename(args.co) + ".codeobject.elf")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    open(out, "wb").write(blob)
    print(f"extracted         : {out}")

    rc = report(blob, args.kernel)
    if rc == 0 and args.load_check:
        rc = load_check(out, args.kernel)
    return rc


if __name__ == "__main__":
    sys.exit(main())
