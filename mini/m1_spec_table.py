"""M1: accelerator specification table.

Transcribes published per-chip figures from vendor documentation and derives
machine balance, the arithmetic intensity a kernel must exceed to be
compute-bound on each device.

Requires no accelerator. The figures live in `tpuperf/specs.py`; this script
formats them and records the derived quantities.

Run:
    python mini/m1_spec_table.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tpuperf.specs import GB, SPECS  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
DTYPE = "bf16"


def main() -> None:
    header = (
        f"{'device':<22}{'peak':>9}{'dtype':>7}{'balance':>9}"
        f"{'bandwidth':>13}{'memory':>10}  other precisions"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for key, s in SPECS.items():
        # Devices without a bf16 path are reported against their own best
        # dtype; a bf16 balance for them would describe an emulation.
        dtype = DTYPE if s.supports(DTYPE) else s.reference_dtype
        peak = s.peak(dtype)
        balance = s.machine_balance(dtype)
        others = ", ".join(
            f"{d} {v:g}" for d, v in s.peak_tflops.items() if d != dtype
        )
        print(
            f"{s.name:<22}{peak:>9.1f}{dtype:>7}{balance:>9.1f}"
            f"{s.bandwidth_as_documented:>13}{s.capacity_as_documented:>10}"
            f"  {others}"
        )
        rows.append(
            {
                "key": key,
                "name": s.name,
                "vendor": s.vendor,
                "reference_dtype": dtype,
                "supports_bf16": s.supports(DTYPE),
                "peak_tflops": s.peak_tflops,
                "machine_balance_flop_per_byte": round(balance, 2),
                "memory_bandwidth_gb_per_s": s.memory_bandwidth_bytes_per_s / GB,
                "memory_kind": s.memory_kind,
                "bandwidth_as_documented": s.bandwidth_as_documented,
                "capacity_as_documented": s.capacity_as_documented,
                "mxu_dim": s.mxu_dim,
                "tensorcores": s.tensorcores,
                "pod_chips": s.pod_chips,
                "notes": s.notes,
                "source": s.source,
            }
        )

    tpus = {k: s for k, s in SPECS.items() if s.vendor == "Google"}
    balances = {k: s.machine_balance(DTYPE) for k, s in tpus.items()}
    lo, hi = min(balances, key=balances.get), max(balances, key=balances.get)
    spread = balances[hi] / balances[lo]

    print()
    print(f"TPU machine balance ({DTYPE}) spans {balances[lo]:.0f} ({lo}) to "
          f"{balances[hi]:.0f} ({hi}), a factor of {spread:.1f}")

    # v5e bandwidth is published in GiB/s while the other generations use GB/s.
    # The reading changes machine balance by about 7%.
    v5e = SPECS["v5e"]
    as_si = v5e.peak(DTYPE) * 1e12 / (800 * GB)
    print(f"v5e unit ambiguity: {v5e.machine_balance(DTYPE):.1f} "
          f"(GiBps, as documented) vs {as_si:.1f} (GBps)")

    no_bf16 = [s.name for s in SPECS.values() if not s.supports(DTYPE)]
    if no_bf16:
        print(f"No published bf16 path: {', '.join(no_bf16)}")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "experiment": "m1_spec_table",
        "note": (
            "Transcribed from vendor documentation. Peak figures are "
            "theoretical; no bandwidth was measured."
        ),
        "devices": rows,
        "derived": {
            "tpu_machine_balance_min": {lo: round(balances[lo], 2)},
            "tpu_machine_balance_max": {hi: round(balances[hi], 2)},
            "tpu_machine_balance_spread": round(spread, 2),
            "v5e_balance_gib_reading": round(v5e.machine_balance(DTYPE), 2),
            "v5e_balance_gb_reading": round(as_si, 2),
            "devices_without_bf16": no_bf16,
        },
    }
    path = RESULTS / "m1_spec_table.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
