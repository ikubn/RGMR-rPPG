#!/usr/bin/env python3
"""Build VIPL motion-transfer pair manifests.

A (replace): source=v2 (motion), driving=v1 (stable)
B (augment): source=v1 (stable), driving=v2 (motion)
C (mobile->stable): source=v9 (phone/mobile motion), driving=v1 (stable)

Both manifests enforce:
- no source4
- no entries listed in NIR.txt
- same subject + same source
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


REL_DIR_RE = re.compile(r"^(p\d+)/(v[0-9]+(?:-[0-9]+)?)/(source\d+)$")


def normalize_rel_dir(path_like: str) -> str | None:
    text = str(path_like).strip().replace("\\", "/")
    if not text:
        return None
    text = text.strip("/")
    if text.startswith("data/"):
        text = text[len("data/") :]
    m = REL_DIR_RE.match(text)
    if not m:
        return None
    return f"{m.group(1)}/{m.group(2)}/{m.group(3)}"


def load_nir_set(nir_path: Path) -> Set[str]:
    if not nir_path.exists():
        raise FileNotFoundError(f"Missing NIR list: {nir_path}")
    out: Set[str] = set()
    with nir_path.open("r", encoding="utf-8") as f:
        for line in f:
            rel = normalize_rel_dir(line)
            if rel:
                out.add(rel)
    return out


def resolve_data_root(vipl_root: Path) -> Path:
    vipl_root = vipl_root.resolve()
    if (vipl_root / "data").is_dir():
        return (vipl_root / "data").resolve()
    return vipl_root


def scan_vipl_records(
    data_root: Path,
    nir_set: Set[str],
    scenarios: Iterable[str],
    exclude_source4: bool = True,
    exclude_nir: bool = True,
) -> Dict[Tuple[int, int, str], str]:
    """Return map: (subject, source_id, scenario) -> rel_dir."""
    records: Dict[Tuple[int, int, str], str] = {}
    for subj_dir in sorted(data_root.glob("p*")):
        if not subj_dir.is_dir():
            continue
        subj_name = subj_dir.name
        if not subj_name.startswith("p"):
            continue
        try:
            subject = int(subj_name[1:])
        except ValueError:
            continue

        for scenario in scenarios:
            scenario_dir = subj_dir / scenario
            if not scenario_dir.is_dir():
                continue
            for src_dir in sorted(scenario_dir.glob("source*")):
                if not src_dir.is_dir():
                    continue
                src_name = src_dir.name
                if not src_name.startswith("source"):
                    continue
                try:
                    source_id = int(src_name.replace("source", ""))
                except ValueError:
                    continue
                if exclude_source4 and source_id == 4:
                    continue

                rel_dir = src_dir.relative_to(data_root).as_posix()
                if exclude_nir and rel_dir in nir_set:
                    continue

                video_path = src_dir / "video.avi"
                if not video_path.exists():
                    continue

                key = (subject, source_id, scenario)
                records[key] = rel_dir
    return records


def build_pairs(records: Dict[Tuple[int, int, str], str]) -> Tuple[List[dict], List[dict], List[dict]]:
    rows_a: List[dict] = []
    rows_b: List[dict] = []
    rows_c: List[dict] = []

    grouped = defaultdict(dict)
    for (subject, source_id, scenario), rel_dir in records.items():
        grouped[(subject, source_id)][scenario] = rel_dir

    for (subject, source_id), per_scenario in sorted(grouped.items()):
        rel_v1 = per_scenario.get("v1")
        rel_v2 = per_scenario.get("v2")
        if not rel_v1 or not rel_v2:
            continue

        # A: motion -> stable, keep source labels from v2
        rows_a.append(
            {
                "source_path": rel_v2,
                "driving_path": rel_v1,
                "subject": subject,
                "source_id": source_id,
                "source_name": f"source{source_id}",
                "source_scenario": "v2",
                "driving_scenario": "v1",
                "group": "A_replace_v2_to_v1",
            }
        )

        # B: stable -> motion, keep source labels from v1
        rows_b.append(
            {
                "source_path": rel_v1,
                "driving_path": rel_v2,
                "subject": subject,
                "source_id": source_id,
                "source_name": f"source{source_id}",
                "source_scenario": "v1",
                "driving_scenario": "v2",
                "group": "B_augment_v1_to_v2",
            }
        )

        rel_v9 = per_scenario.get("v9")
        if rel_v1 and rel_v9:
            rows_c.append(
                {
                    "source_path": rel_v9,
                    "driving_path": rel_v1,
                    "subject": subject,
                    "source_id": source_id,
                    "source_name": f"source{source_id}",
                    "source_scenario": "v9",
                    "driving_scenario": "v1",
                    "group": "C_replace_v9_to_v1",
                }
            )

    return rows_a, rows_b, rows_c


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    fields = [
        "source_path",
        "driving_path",
        "subject",
        "source_id",
        "source_name",
        "source_scenario",
        "driving_scenario",
        "group",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def validate_rows(
    name: str,
    rows: List[dict],
    data_root: Path,
    nir_set: Set[str],
    expected_count: int | None,
) -> List[str]:
    errs: List[str] = []
    seen = set()
    dup = 0
    missing = 0
    in_nir = 0
    uses_source4 = 0
    source_counter = Counter()

    for row in rows:
        key = (row["source_path"], row["driving_path"], int(row["subject"]), int(row["source_id"]))
        if key in seen:
            dup += 1
        else:
            seen.add(key)

        src_rel = normalize_rel_dir(row["source_path"])
        drv_rel = normalize_rel_dir(row["driving_path"])
        if src_rel is None or drv_rel is None:
            missing += 1
            continue

        src_abs = data_root / src_rel
        drv_abs = data_root / drv_rel
        if not (src_abs / "video.avi").exists() or not (drv_abs / "video.avi").exists():
            missing += 1

        if src_rel in nir_set or drv_rel in nir_set:
            in_nir += 1

        if src_rel.endswith("/source4") or drv_rel.endswith("/source4"):
            uses_source4 += 1

        source_counter[row["source_name"]] += 1

    if not rows:
        errs.append(f"{name}: no rows")
    if dup:
        errs.append(f"{name}: duplicate rows={dup}")
    if missing:
        errs.append(f"{name}: rows with missing source/driving video={missing}")
    if in_nir:
        errs.append(f"{name}: rows touching NIR entries={in_nir}")
    if uses_source4:
        errs.append(f"{name}: rows touching source4={uses_source4}")
    if expected_count is not None and len(rows) != expected_count:
        errs.append(f"{name}: row_count={len(rows)} expected={expected_count}")

    print(f"{name}: rows={len(rows)} by_source={dict(source_counter)}")
    return errs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build VIPL pair manifests (v1/v2/v9, noNIR/source4).")
    parser.add_argument("--vipl-root", default="datasets/VIPL-HR-V1", help="VIPL root (contains data/ and NIR.txt) or data/ dir.")
    parser.add_argument("--nir-list", default="datasets/VIPL-HR-V1/NIR.txt", help="Path to VIPL NIR.txt")
    parser.add_argument("--out-dir", default="manifests/vipl_cv", help="Output directory for pair CSVs")
    parser.add_argument("--expected-a", type=int, default=None, help="Optional expected row count for A manifest")
    parser.add_argument("--expected-b", type=int, default=None, help="Optional expected row count for B manifest")
    parser.add_argument("--expected-c", type=int, default=None, help="Optional expected row count for C manifest")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if validation fails")
    args = parser.parse_args()

    vipl_root = Path(args.vipl_root).resolve()
    nir_list = Path(args.nir_list).resolve()
    out_dir = Path(args.out_dir).resolve()

    data_root = resolve_data_root(vipl_root)
    if not data_root.exists():
        raise FileNotFoundError(f"VIPL data root not found: {data_root}")

    nir_set = load_nir_set(nir_list)
    records = scan_vipl_records(
        data_root,
        nir_set,
        scenarios=("v1", "v2", "v9"),
        exclude_source4=True,
        exclude_nir=True,
    )
    rows_a, rows_b, rows_c = build_pairs(records)

    out_a = out_dir / "vipl_pairs_A_v2_to_v1.csv"
    out_b = out_dir / "vipl_pairs_B_v1_to_v2.csv"
    out_c = out_dir / "vipl_pairs_C_v9_to_v1.csv"
    out_summary = out_dir / "vipl_pairs_summary.json"

    write_csv(out_a, rows_a)
    write_csv(out_b, rows_b)
    write_csv(out_c, rows_c)

    errors: List[str] = []
    errors.extend(validate_rows("A_v2_to_v1", rows_a, data_root, nir_set, args.expected_a))
    errors.extend(validate_rows("B_v1_to_v2", rows_b, data_root, nir_set, args.expected_b))
    errors.extend(validate_rows("C_v9_to_v1", rows_c, data_root, nir_set, args.expected_c))

    summary = {
        "vipl_root": str(vipl_root),
        "data_root": str(data_root),
        "nir_list": str(nir_list),
        "nir_entries": len(nir_set),
        "scanned_records_v1v2v9_after_filter": len(records),
        "pairs_A_v2_to_v1": len(rows_a),
        "pairs_B_v1_to_v2": len(rows_b),
        "pairs_C_v9_to_v1": len(rows_c),
        "out_a": str(out_a),
        "out_b": str(out_b),
        "out_c": str(out_c),
        "validation_errors": errors,
    }
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {out_a}")
    print(f"Wrote: {out_b}")
    print(f"Wrote: {out_c}")
    print(f"Wrote: {out_summary}")
    if errors:
        print("Manifest validation errors:")
        for e in errors:
            print(f"  - {e}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("Manifest validation passed.")


if __name__ == "__main__":
    main()
