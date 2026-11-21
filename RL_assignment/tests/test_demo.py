import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


def test_csv_export_via_command_line(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "dynamics_sanity.py"
    subprocess.run([sys.executable, str(script), "--output-dir", str(tmp_path)], check=True)
    with (tmp_path / "dynamics_trace.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    counts = Counter((r["scenario"], float(r["u"])) for r in rows)
    assert counts == {("cold", 0): 96, ("hot", 0): 96, ("transition", 0): 96,
                      ("transition", -0.5): 8, ("transition", 0.5): 8}
    for row in rows:
        assert float(row["time_h"]) - float(row["time_h_start"]) == 0.25
        assert float(row["energy_kwh_step"]) == pytest.approx(float(row["P_el_kw"]) * 0.25)
    first = rows[0]
    assert float(first["T_i_start_c"]) == 22 and float(first["T_e_start_c"]) == 18
    assert float(first["T_o_c"]) == pytest.approx(float(rows[1]["T_o_applied_c"]), abs=1e-6)
