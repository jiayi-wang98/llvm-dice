# The DICE backend for NVPTX

This branch (`dice-compiler-backend`) adds a **DICE** code-generation path to the
NVPTX target. DICE is a statically-scheduled, CGRA-based SIMT GPGPU that replaces
the SIMD backend of a Turing-class GPU while keeping the CUDA/SIMT programming
model (Wang, Lu, Zeng, Li, *"DICE: Enabling Efficient General-Purpose SIMT
Execution with Statically Scheduled Coarse-Grained Reconfigurable Arrays"*,
ISCA 2026, [arXiv:2605.05496](https://arxiv.org/abs/2605.05496)).

Nothing outside `llvm/lib/Target/NVPTX/` is touched. The delta is six new files
plus modifications to nine pre-existing ones; `git diff <base> HEAD --stat` shows
the whole of it.

```
git clone -b dice-compiler-backend git@github.com:jiayi-wang98/llvm-dice.git
```

is a complete, self-sufficient entry point for compiler work: everything in this
document is done with this tree, a CUDA installation, cmake and ninja. The rest
of the DICE stack (mapper, bitstream generation, fabric RTL, verification loops)
lives in **DICE-IDE**, `git@github.com:jiayi-wang98/DICE-IDE.git`; see
[Where the rest of the stack is](#where-the-rest-of-the-stack-is).

Every command below was run on 2026-08-11 against a `clang++` built from this
branch, and its real output is shown. The two exceptions are marked
**UNVERIFIED** with the reason.

---

## 1. What it compiles, and what it emits

DICE executes straight-line dataflow regions called **p-graphs**, each mapped
spatially onto the CGRA: every instruction of a p-graph occupies its own
functional unit, a load written inside a p-graph is only readable in a *later*
one, and a barrier can only stand at a p-graph's head. A p-graph is configured
from a bitstream, which costs roughly 360 cycles to fetch, so how the compiler
cuts the program into p-graphs is the dominant performance decision.

The backend therefore does two things stock NVPTX does not:

1. **Partitioning** — it splits machine basic blocks until *every* basic block is
   a legal p-graph, and forces a label on each one, so the emitted PTX's labels
   are exactly the p-graph boundaries. This form is called **partitioned PTX**
   (`.pptx` by convention downstream).
2. **Architectural register assignment** — NVPTX normally emits unbounded
   virtual registers. DICE has a real register file, so the backend assigns
   `%r` (general, pool of 32), `%c` (parameter-load destinations), `%p`
   (predicates), and `%w` (wires — values whose whole life is inside one
   p-graph). The indices are load-bearing, not cosmetic: the register file banks
   as `(tid + reg) % 32`.

Both are **off by default**. Stock NVPTX behaviour is unchanged unless
`-nvptx-dice-partition` is passed; `nvptxDiceEnabled()`
(`NVPTXDicePartition.cpp`) returns that flag and every DICE pass tests it.

Output is a single `.ptx` file that carries two extra kinds of comment:

| Comment | Emitted by | Meaning |
|---|---|---|
| `// DICE_META …` | `NVPTXAsmPrinter::emitFunctionBodyEnd` | one record per p-graph: id, bitstream address/length, latency, in/out registers, branch info. Split out into a `.meta` sidecar downstream |
| `// DICE_QUAL …` | same | a *semantic qualification* — a place where the emitted code is not bit-identical to what PTX asked for. Today only fp64 FMA lowering (§6) produces one |
| `// DICE_PRED @[!]%pN` | `NVPTXAsmPrinter::emitInstruction` | an if-conversion guard on the preceding instruction. PTX has no MC-level predication, so the guard is emitted as a trailing marker and rewritten into the `@%p` prefix by the driver |

## 2. Build

The authoritative configuration is DICE-IDE's `deps/pins.ini`, section
`[llvm-build]`: Ninja, `Release`, `LLVM_ENABLE_PROJECTS=clang`,
`LLVM_TARGETS_TO_BUILD=X86;NVPTX`, assertions `OFF`.

```bash
cmake -G Ninja -S llvm -B ../llvm-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_PROJECTS=clang \
      -DLLVM_TARGETS_TO_BUILD='X86;NVPTX' \
      -DLLVM_ENABLE_ASSERTIONS=OFF
ninja -C ../llvm-build -j16 clang llc opt
```

**UNVERIFIED — not re-run here.** These are the pinned commands (DICE-IDE's
`tools/dice-bootstrap llvm-build` issues exactly them, `tools/dice-bootstrap`
lines 331–338), and they are what produced the binary used throughout this
document; but a rebuild is tens of minutes (`tools/dice-bootstrap`'s own usage
text calls it "the 90-minute one") and was deliberately not repeated. The build
directory on
the reference machine was additionally configured with
`-DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF`, which only shortens
the build.

NVPTX is the DICE target; X86 is the host target of the machine this was pinned
on. Substitute your host architecture if it differs, but keep NVPTX.

Confirm the DICE passes are in the binary you built:

```console
$ clang++ -cc1 -mllvm -help-hidden | grep -c nvptx-dice
15
```

That is the check DICE-IDE's `dice-bootstrap verify` performs on `llc` — a
binary built from a tree without this delta answers `0`, and `clang --version`
cannot tell you, because LLVM stamps the revision the worktree had at *configure*
time, which is the upstream base rather than this branch.

## 3. A worked example

`saxpy.cu`:

```cuda
extern "C" __global__ void saxpy(int n, float a, const float *x, float *y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    y[i] = a * x[i] + y[i];
}
```

Compile it for DICE (adjust `--cuda-path` and the `clang++` path):

```bash
clang++ -x cuda --cuda-device-only --cuda-gpu-arch=sm_52 \
        --cuda-path=/usr/local/cuda-11.7 -O2 -S \
        -mllvm -nvptx-no-bfe \
        -mllvm -nvptx-dice-partition \
        -mllvm -nvptx-dice-regalloc \
        -mllvm -nvptx-dice-gpr-count=32 \
        -mllvm -nvptx-dice-pred-count=16 \
        -mllvm -nvptx-dice-pe-count=16 \
        -mllvm -nvptx-dice-sfu-count=4 \
        -mllvm -nvptx-dice-ldst-count=4 \
        -mllvm -nvptx-dice-imm-bits=16 \
        -o saxpy.ptx saxpy.cu
```

The six budget/pool flags are the `dice_v2` device's values; §5 explains why you
should always pass them rather than take the defaults. Verified output, complete
and unedited apart from the PTX header:

```ptx
.visible .entry saxpy(
	.param .u32 saxpy_param_0,
	.param .f32 saxpy_param_1,
	.param .u64 .ptr .align 1 saxpy_param_2,
	.param .u64 .ptr .align 1 saxpy_param_3
)
{
	.reg .b32 	%r<4>;
	.reg .b32 	%c<5>;
	.reg .pred 	%p<2>;
	.reg .b32 	%w<5>;

// %bb.0:
	ld.param.b32 	%c1, [saxpy_param_0];
	ld.param.b32 	%c0, [saxpy_param_1];
	ld.param.b64 	%c2, [saxpy_param_2];
	ld.param.b64 	%c3, [saxpy_param_3];
$L__BB0_1:                              // Label of block must be emitted
	mov.u32 	%w0, %ctaid.x;
	mov.u32 	%w1, %ntid.x;
	mov.u32 	%w2, %tid.x;
	mad.lo.s32 	%r0, %w0, %w1, %w2;
	setp.ge.s32 	%p0, %r0, %c1;
	@%p0 bra 	$L__BB0_4;
$L__BB0_2:                              // Label of block must be emitted
	cvta.to.global.u64 	%w0, %c2;
	cvta.to.global.u64 	%w1, %c3;
	mul.wide.s32 	%w2, %r0, 4;
	add.s64 	%w3, %w0, %w2;
	ld.global.b32 	%r0, [%w3];
	add.s64 	%r1, %w1, %w2;
	ld.global.b32 	%r2, [%r1];
$L__BB0_3:                              // Label of block must be emitted
	fma.rn.f32 	%w0, %c0, %r0, %r2;
	st.global.b32 	[%r1], %w0;
$L__BB0_4:                              // Label of block must be emitted
	ret;
// DICE_META FUNCTION = saxpy;
// DICE_META 
// DICE_META DBB_ID = 0,
// DICE_META BITSTREAM_ADDR = $DICE_BB0_0,
// DICE_META BITSTREAM_LENGTH = 4,
// DICE_META UNROLLING_FACTOR = 1,
// DICE_META UNROLLING_STRATEGY = 0,
// DICE_META LAT = 1,
// DICE_META LD_DEST_REGS = (%c0, %c1, %c2, %c3),
// DICE_META IS_PARAMETER_LOAD;
// DICE_META 
// DICE_META // SR_IN = (%ctaid.x, %ntid.x, %tid.x)
// DICE_META DBB_ID = 1,
// DICE_META BITSTREAM_ADDR = $DICE_BB0_1,
// DICE_META BITSTREAM_LENGTH = 6,
// DICE_META UNROLLING_FACTOR = 1,
// DICE_META UNROLLING_STRATEGY = 0,
// DICE_META LAT = 4,
// DICE_META IN_REGS = (%c1),
// DICE_META OUT_REGS = (%p0, %r0),
// DICE_META BRANCH = 1,
// DICE_META BRANCH_PRED = (%p0),
// DICE_META BRANCH_TARGET = 4,
// DICE_META BRANCH_RECVPC = 4;
// DICE_META 
// DICE_META DBB_ID = 2,
// DICE_META BITSTREAM_ADDR = $DICE_BB0_2,
// DICE_META BITSTREAM_LENGTH = 7,
// DICE_META UNROLLING_FACTOR = 1,
// DICE_META UNROLLING_STRATEGY = 0,
// DICE_META LAT = 4,
// DICE_META IN_REGS = (%c2, %c3, %r0),
// DICE_META OUT_REGS = (%r1),
// DICE_META LD_DEST_REGS = (%r0, %r2);
// DICE_META 
// DICE_META DBB_ID = 3,
// DICE_META BITSTREAM_ADDR = $DICE_BB0_3,
// DICE_META BITSTREAM_LENGTH = 2,
// DICE_META UNROLLING_FACTOR = 1,
// DICE_META UNROLLING_STRATEGY = 0,
// DICE_META LAT = 3,
// DICE_META IN_REGS = (%c0, %r0, %r1, %r2),
// DICE_META STORE = 1;
// DICE_META 
// DICE_META DBB_ID = 4,
// DICE_META BITSTREAM_ADDR = $DICE_BB0_4,
// DICE_META RET;
```

Read it as five p-graphs:

- **DBB 0** — the parameter load. `ld.param` costs no LDST port and its results
  *are* readable, so all four land in one dispatch-once block, into `%c`.
- **DBB 1** — index computation, terminated by the guarded branch. `%w0..%w2`
  are wires: they never leave the block. `%r0` does, so it is a `%r`.
- **DBB 2** — the two loads. The partitioner had to cut *before* the `fma`,
  because `fma` uses `%r0`/`%r2`, which the LDST unit only writes back at the
  region's exit (partition rule 2).
- **DBB 3** — the multiply-add and the store.
- **DBB 4** — the return.

The same compilation without `-nvptx-dice-regalloc` was measured to emit **0**
`DICE_META` lines and stock `%r`/`%rd` virtual numbering: partitioning alone
gives p-graph boundaries but no register assignment and no metadata. Both flags
are needed for a usable artifact.

**Use `-O1` or higher.** The DICE IR-level passes hook `addIRPasses()`, which
NVPTX only reaches when the opt level is not `None`, so at `-O0` neither
vector-memop scalarization nor sub-word widening runs. Measured on a kernel
loading `unsigned char`: at `-O2` the backend emits
`ld.global.b32` + `and.b32 …, 255`, which the fabric can execute; at `-O0` it
emits `ld.b8`, which the fabric **cannot express at all** (there is no
access-size field and no byte enable — a load returns the four bytes at its byte
address). `-O0` also leaves local-memory traffic that DICE has no path for.

### Splitting the two artifacts

DICE-IDE's `tools/dicc` is the reference splitter, and downstream tools expect
its exact output. It (`split_native()`) moves every `// DICE_META` line into a
`.meta` file, leaves the rest as the `.pptx`, and applies four text fixups:

1. `$L__BB<f>_<n>` → `$DICE_BB<f>_<n>` — the label prefix every downstream tool
   greps for, and what `BITSTREAM_ADDR` already names;
2. an explicit `$DICE_BB<f>_0:` label at the top of each function body, because
   the asm printer never labels an entry block but DBB 0 needs an address;
3. `inst; // DICE_PRED @!%pN` → `@!%pN inst;` — the guard folded into the PTX
   prefix form;
4. untyped `.b8/.b16/.b32/.b64` on `ld`/`ldu`/`st`, and on `setp.eq/ne`, respelled
   as `.u…` — the simulator's DICE mode hangs on untyped-`.b` memory ops, and
   `eq`/`ne` on `.b` is sign-blind, so the spellings are equivalent.

If you are working only in this tree, the raw `.ptx` above is the compiler's real
output and is what you should read and diff. If you are producing artifacts for
the rest of the stack, run `dicc` rather than reimplementing those fixups.

## 4. The passes, and the order they run in

Registered in `NVPTXTargetMachine.cpp`. **Read that file rather than this list if
they ever disagree.**

In `NVPTXPassConfig::addIRPasses()`, only when `nvptxDiceEnabled()` and the opt
level is not `None`, replacing the load/store vectorizer:

| # | Pass | File | What it does |
|---|---|---|---|
| 1 | `ScalarizerPass` (upstream, `ScalarizeLoadStore=true`) | — | DICE has no vector LDST: a `v4` load is four ports, a whole p-graph's budget. Rather than form vector memops and split them again, they are never formed, and the ones the *source* wrote (e.g. Parboil's `float4`) are scalarized |
| 2 | `NVPTXDiceWidenSubword` | `NVPTXDiceWidenSubword.cpp` | 8/16-bit **loads** → a 32-bit load plus an explicit mask, and rewrites that value's `zext`/`sext` into `and` / `shl`+`ashr` — forms the PE's ALU can execute (a `cvt.s32.s8` is not in the fabric's op table). Must precede the following InstCombine. Sub-word *stores* are deliberately left alone; on `-nvptx-dice-widen-subword=false` the loads come out reading their neighbours' bytes |

Then in `NVPTXPassConfig::addPreEmitPass()`, in this order — after PHI
elimination, and with virtual registers still live, since NVPTX has no register
allocator and instructions map 1:1 onto emitted PTX:

| # | Pass | File | What it does |
|---|---|---|---|
| 3 | `NVPTXDiceFP64Lower` | `NVPTXDiceFP64Lower.cpp` | `fma.rn.f64` → `mul` + `add`. DICE's fp64 unit is two-operand by construction (a PE tile has four 32-bit diagonal inputs; two fp64 operands already fill them), so a fused form would multiply-add a garbage third operand. Runs **before** partitioning because it turns one SFU tile into two and the SFU budget must see that. Records a `DICE_QUAL` because two roundings is not what `.rn` asked for |
| 4 | `NVPTXDiceIfConvert` | `NVPTXDiceIfConvert.cpp` | opt `ifcvt` — small triangles/diamonds → `@%p`-predicated straight-line code. **Default off, and withdrawn; see §6** |
| 5 | `NVPTXDiceLoadSched` | `NVPTXDiceLoadSched.cpp` | opt `loadsched` — hoists loads past instructions they do not depend on so they cluster, and one load-to-use split serves the whole cluster. Default off |
| 6 | `NVPTXDicePartition` | `NVPTXDicePartition.cpp` | the partitioner. Splits blocks until every block is a legal p-graph and labels each. Rules: a barrier heads its p-graph; a use of a load/atomic result issued in the current p-graph starts a new one (param loads exempt); PE/SFU/LDST budgets; terminators already split. Costs: `ld`/`st`/`atom` = 1 LDST (param loads free), div/sqrt-family = 1 SFU, `cvta` and 32↔64 integer `cvt` free, `mov` and special-register reads free, everything else 1 PE. A literal that fits `-nvptx-dice-imm-bits` folds into the tile for free; one that does not must be materialised, and that is a PE cost |
| 7 | `NVPTXDiceFuse` | `NVPTXDicePartition.cpp` (bottom) | opt `fuse` — re-merges adjacent p-graphs when the merged block is still legal. Default off |
| 8 | `NVPTXDiceRegAlloc` | `NVPTXDiceRegAlloc.cpp` | assigns `%r`/`%c`/`%p`/`%w` over block-granularity live intervals (blocks *are* p-graphs at this point) and renders the `.meta` records. The assignment is stored in `NVPTXMachineFunctionInfo` as queryable state, not printed directly, because later decisions depend on real indices |

`NVPTXAsmPrinter` then names registers from that assignment
(`getVirtualRegisterName`, `encodeVirtualRegister`), sizes the `.reg`
declarations from what the allocator actually used, and emits the `DICE_QUAL` and
`DICE_META` comment blocks at the end of each function body.

Six new source files, eight pass instances — `NVPTXDiceFuse` shares
`NVPTXDicePartition.cpp` with the partitioner, and pass 1 is upstream's.

## 5. Flags

All are `cl::Hidden`, so pass them through `-mllvm` from clang (or directly to
`llc`/`opt`). `clang++ -cc1 -mllvm -help-hidden | grep nvptx-dice` lists exactly
these 15.

### Mode

| Flag | Default | Effect |
|---|---|---|
| `-nvptx-dice-partition` | `false` | Split basic blocks into p-graphs and label every block. **This is the master switch**: `nvptxDiceEnabled()` is this flag, so it also gates the scalarizer swap, `NVPTXDiceWidenSubword`, and `NVPTXDiceFP64Lower` |
| `-nvptx-dice-regalloc` | `false` | Assign `%r`/`%c`/`%p`/`%w` and render the `DICE_META` records |
| `-nvptx-no-bfe` | `false` | Do not select `BFE`; emit shifts and masks instead. Not DICE-specific in name, but added by this branch and passed by every DICE driver — the fabric has no bit-field unit |

### Hardware description — these must match your device

| Flag | `cl::opt` default | `dice_v2` value | Meaning |
|---|---|---|---|
| `-nvptx-dice-gpr-count` | `32` | 32 | `%r` pool size |
| `-nvptx-dice-pred-count` | **`24`** | **16** | `%p` pool size |
| `-nvptx-dice-pe-count` | `16` | 16 | PE ops per p-graph |
| `-nvptx-dice-sfu-count` | `4` | 4 | SFU ops per p-graph |
| `-nvptx-dice-ldst-count` | `4` | 4 | LDST accesses per p-graph |
| `-nvptx-dice-imm-bits` | `16` | 16 | width of the *signed* immediate a PE tile's configuration holds; `0` = none |
| `-nvptx-dice-const-reads` | `0` | *(unset)* | distinct constant-file reads per p-graph; `0` = unenforced. The RF→CGRA path is a full 60×N crossbar (32 banks + 16 const + 12 special), so there is no limit today |

> **The defaults are not the hardware.** They are a *second, independent*
> declaration of the device, and one of them disagrees with the real one:
> `-nvptx-dice-pred-count` defaults to **24** while `dice_v2`'s register file has
> **16** predicate registers. Drive clang yourself and you are compiling for a
> machine with eight predicate registers that do not exist. Consequences,
> measured in DICE-IDE (see the docstring of `device_backend_flags` in
> `tools/dicc`): the allocator's `colorIntervals` is first-fit and takes the
> lowest free colour, no kernel in the 22-kernel corpus ever selected `%p16` or
> above, and all 22 recompile byte-identically at 16 — so today the difference is
> latent rather than active. But a kernel that really needs a 17th predicate
> compiles happily at the default and is then refused *by name* much later, by
> `dice-verify` check C5 and by `dice-pack`'s register bitmap; at 16 it fails
> inside the allocator instead ("DICE regalloc: %p pool exhausted in <fn>"),
> which is earlier and cannot produce an illegal artifact.
>
> `tools/dicc` exists partly to close this hole: it reads `device/dice_v2.yaml`
> and forwards all six values as `-mllvm -nvptx-dice-*` flags, and refuses to
> compile at all if it cannot read the device file. If you invoke clang directly,
> **pass the six flags explicitly**, as §3 does. A repeated `cl::opt` takes its
> last value, so a later flag overrides an earlier one.

### Optimizations and diagnostics

| Flag | Default | Effect |
|---|---|---|
| `-nvptx-dice-opt-ifcvt` | `false` | enable if-conversion. **Withdrawn — §6** |
| `-nvptx-dice-ifcvt-max-ops` | `12` | max real instructions per converted side block |
| `-nvptx-dice-opt-fuse` | `false` | enable p-graph fusion |
| `-nvptx-dice-opt-loadsched` | `false` | enable load clustering |
| `-nvptx-dice-widen-subword` | **`true`** | widen 8/16-bit loads. On by default whenever the DICE pipeline is on. Turning it off emits loads whose upper bytes are the neighbouring data — it is a debugging knob, not a tuning one |
| `-nvptx-dice-cost-dump` | `false` | print each p-graph's tile accounting to stderr |

`-nvptx-dice-cost-dump` is the tool for arguing with the partitioner. Verified on
the §3 kernel:

```console
$ clang++ … -mllvm -nvptx-dice-partition -mllvm -nvptx-dice-regalloc \
           -mllvm -nvptx-dice-cost-dump -o /dev/null saxpy.cu
DICE-COST saxpy 1 +PE=0 (op=0 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=0) cvta_to_global_64
DICE-COST saxpy 1 +PE=0 (op=0 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=0) cvta_to_global_64
DICE-COST saxpy 1 +PE=1 (op=1 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=1) MUL_WIDEs32_ri
DICE-COST saxpy 1 +PE=1 (op=1 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=2) ADD64rr
DICE-COST saxpy 1 +PE=0 (op=0 disp=0 mat=0) +SFU=0 +LDST=1 running(PE=2) LD_i32
DICE-COST saxpy 1 +PE=1 (op=1 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=3) ADD64rr
DICE-COST saxpy 1 +PE=0 (op=0 disp=0 mat=0) +SFU=0 +LDST=1 running(PE=3) LD_i32
DICE-COST saxpy 1 SPLIT before FMA_F32rrr (would be PE=4 of 16, SFU=0, LDST=2, lateDef=1, barrier=0)
DICE-COST saxpy 4 +PE=1 (op=1 disp=0 mat=0) +SFU=0 +LDST=0 running(PE=1) FMA_F32rrr
…
```

The `SPLIT` line names the instruction that closed the p-graph and why — here
`lateDef=1`, the load-to-use hazard, not a budget. Three programs implement this
cost model (this pass, DICE-IDE's `dicemap.pgraph`, and `dice-verify` check C4)
and they are required to agree; this dump is how a disagreement is localised to
one instruction.

## 6. The three optimizations are off, and one is withdrawn

The flagless pipeline is the **certified baseline**. DICE-IDE freezes its output
byte-for-byte (`docs/baseline_naive.json`) precisely so that any change here is
detectable, and each optimization is a separate flag so that each is
individually attributable and bisectable. Do not assume any of them is a
default.

Status, from DICE-IDE's `docs/opt_results.md`:

- **`loadsched`** — certified; the only surviving measured saving, **−2.1%**.
- **`fuse`** — certified, and an identity: it produces artifacts
  byte-identical to the baseline, because the partitioner already packs
  maximally. Its value was compositional, in combination with `ifcvt`.
- **`ifcvt`** — **WITHDRAWN 2026-08-09**, along with `ifcvt,fuse` and
  `ifcvt,fuse,loadsched`. The measured numbers (up to −13.0%) are retained in
  that document as a record of what withdrawal costs, and are *not* a
  certification.

Why `ifcvt` is withdrawn, in one paragraph, from
`docs/review/05-triage-T3-predication.md`: **a predicated write to an
architectural register has no representation below the PTX.** The hardware's
`pgraph_meta_t` carries `out_regs_bitmap`, one present-or-absent bit per
register — no per-register write enable, no guard field, no per-port store
enable. gpgpu-sim honours `@%p` natively, so the cycle numbers were gated on a
model more capable than the fabric. `dice-verify` check C21 now refuses a
predicated write as a hard error; it fires on **0 of the 817** baseline p-graphs
and on 14 of 27 kernels under `ifcvt` — every kernel the pass measurably helped —
including one *measured* wrong answer (streamcluster `DICE_BB1_13`). So the pass
still exists, still works as designed, and its output is refused downstream.
Restoring it needs either a hardware mechanism or re-certification in the RTL
loop rather than in gpgpu-sim.

Guarded *branches* are unaffected: the terminator's predicate travels in
`branch_meta_t`, which is the frontend's own mechanism, and the §3 example's
`@%p0 bra` is baseline output.

## 7. Where the rest of the stack is

This tree ends at `.pptx` + `.meta`. Everything that consumes them is in
**DICE-IDE** (`git@github.com:jiayi-wang98/DICE-IDE.git`), which is a few MB and
does not vendor LLVM:

| Want | There |
|---|---|
| one command, CUDA → `.pptx` + `.meta`, device-driven and legality-gated | `tools/dicc` |
| is this artifact legal for the device? | `tools/dice-verify/dice-verify` (C1…C21) |
| place & route onto the CGRA, visualise it | `tools/dice-mapper/dice-map`, `tools/dice-viz` |
| the normative device parameters | `device/dice_v2.yaml`, `docs/device-contract.md` |
| what `.meta` fields mean, reconciled across paper / compiler / simulator / RTL | `docs/artifact-formats.md` |
| the exact upstream revisions and build config | `deps/pins.ini` |
| a real, tested clang command line over a benchmark corpus | `tools/dice-crosscheck` (`CLANG_FLAGS`) |
| optimization measurements and withdrawals | `docs/opt_results.md` |
| how to run all of it | `RUNNING.md` |

### If you change this directory, mirror it

DICE-IDE vendors this delta at `llvm/dice/`: the new files verbatim in
`llvm/dice/new/`, the nine modified files as `llvm/dice/upstream.patch`, against
the base commit pinned in `deps/pins.ini`. `tools/dice-bootstrap-tests/`
reconstructs the branch from base + delta and diffs the **whole**
`llvm/lib/Target/NVPTX` subtree, so *any* new file here — including this README,
which is vendored as `llvm/dice/new/README-DICE.md` — must be added there too, or
the two routes to the backend diverge and that test fails. Keep the copies
byte-identical. `deps/pins.ini`'s `local_branch_commit` records this branch's tip
and `dice-bootstrap verify`'s `llvm-fork-refs` row compares it to the pushed
fork, so a local commit that is not pushed is reported rather than silently
shipping two different compilers.

Nothing outside `llvm/lib/Target/NVPTX/` may be modified. If a change needs
`llvm/lib/CodeGen` or `clang/`, the pin stops describing the build and the
vendoring test fails — correctly.
