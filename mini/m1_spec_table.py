"""M1: TPU specification table.

Transcribes published per-chip figures from the Cloud TPU documentation and
derives machine balance, the arithmetic intensity a kernel must exceed to be
compute-bound on each generation.

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

from tpuperf.specs import GB, GIB, SPECS  # noqa: E402

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    header = (
        f"{'generation':<20}{'bf16':>7}{'balance':>10}"
        f"{'bandwidth':>14}{'HBM':>10}{'MXU':>6}{'cores':>7}  low precision"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for key, s in SPECS.items():
        speedup = s.low_precision_speedup
        low = f"{speedup:.0f}x {s.low_precision_format}" if speedup else "-"
        print(
            f"{s.name:<20}{s.bf16_tflops:>7.0f}"
            f"{s.machine_balance:>10.1f}"
            f"{s.bandwidth_as_documented:>14}"
            f"{s.capacity_as_documented:>10}"
            f"{s.mxu_dim:>6}{s.tensorcores:>7}  {low}"
        )
        rows.append(
            {
                "key": key,
                "name": s.name,
                "bf16_tflops": s.bf16_tflops,
                "hbm_bandwidth_gb_per_s": s.hbm_bandwidth_bytes_per_s / GB,
                "hbm_capacity_bytes": s.hbm_capacity_bytes,
                "hbm_capacity_as_documented": s.capacity_as_documented,
                "machine_balance_flop_per_byte": round(s.machine_balance, 2),
                "mxu_dim": s.mxu_dim,
                "tensorcores": s.tensorcores,
                "low_precision_format": s.low_precision_format,
                "low_precision_speedup": speedup,
                "pod_chips": s.pod_chips,
                "bandwidth_as_documented": s.bandwidth_as_documented,
                "source": s.source,
            }
        )

    balances = [s.machine_balance for s in SPECS.values()]
    spread = max(balances) / min(balances)
    print()
    print(f"machine balance spans {min(balances):.0f} to {max(balances):.0f} "
          f"FLOP/byte, a factor of {spread:.1f}")

    # v5e bandwidth is published in GiB/s while the other generations use GB/s.
    # The reading changes machine balance by about 7%.
    v5e = SPECS["v5e"]
    as_si = v5e.bf16_tflops * 1e12 / (800 * GB)
    print(f"v5e unit ambiguity: {v5e.machine_balance:.1f} (GiB/s, as documented) "
          f"vs {as_si:.1f} (GB/s)")

    RESULTS.mkdir(exist_ok=True)
    payload = {
        "experiment": "m1_spec_table",
        "note": (
            "Transcribed from official Cloud TPU documentation. Peak figures "
            "are theoretical; no bandwidth was measured."
        ),
        "chips": rows,
        "derived": {
            "machine_balance_min": round(min(balances), 2),
            "machine_balance_max": round(max(balances), 2),
            "machine_balance_spread": round(spread, 2),
            "v5e_balance_gib_reading": round(v5e.machine_balance, 2),
            "v5e_balance_gb_reading": round(as_si, 2),
        },
    }
    path = RESULTS / "m1_spec_table.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {path.relative_to(RESULTS.parent)}")


if __name__ == "__main__":
    main()
