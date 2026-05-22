#!/usr/bin/env python3
import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from scipy.io import loadmat


SUBJECT_RE = re.compile(r"(?:^|/)subject(\d+)/(?:[^/]+)$")


def normalize_motion(motion):
    if motion == "Stationary (after exercise)":
        return "Stationary"
    if motion == "Watching Videos":
        return "Walking"
    return motion


def normalize_exercise(exercise):
    if exercise == "True":
        return "After-exercise"
    if exercise == "False":
        return "Normal"
    return exercise


def parse_subject_from_relpath(relpath):
    match = SUBJECT_RE.search(relpath)
    if not match:
        return None
    return int(match.group(1))


def load_mmpd_map(root):
    mapping = {}
    for mat_path in sorted(root.glob("subject*/p*_*.mat")):
        meta = loadmat(mat_path, variable_names=["light", "motion", "exercise"])
        light = str(meta["light"].item())
        motion = normalize_motion(str(meta["motion"].item()))
        exercise = normalize_exercise(str(meta["exercise"].item()))
        subject = int(mat_path.parent.name.replace("subject", ""))
        key = (subject, light, motion, exercise)
        mapping[key] = mat_path.relative_to(root).as_posix()
    return mapping


def load_conditions(path, exercise_filter):
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exercise = row["exercise"]
            if exercise_filter and exercise != exercise_filter:
                continue
            rows.append(
                {
                    "subject": int(row["subject"]),
                    "light": row["light"],
                    "motion": row["motion"],
                    "exercise": exercise,
                    "mae": float(row["mae"]),
                }
            )
    return rows


def build_a_w2t(rows, file_map):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["subject"], row["light"], row["exercise"])][row["motion"]] = row

    pairs = []
    missing = 0
    for (subject, light, exercise), motions in sorted(grouped.items()):
        source_row = motions.get("Walking")
        driving_row = motions.get("Talking")
        if source_row is None or driving_row is None:
            continue

        source_key = (subject, light, "Walking", exercise)
        driving_key = (subject, light, "Talking", exercise)
        source_path = file_map.get(source_key)
        driving_path = file_map.get(driving_key)
        if not source_path or not driving_path:
            missing += 1
            continue

        pairs.append(
            {
                "source_path": source_path,
                "driving_path": driving_path,
                "subject": subject,
                "light": light,
                "exercise": exercise,
                "source_motion": "Walking",
                "driving_motion": "Talking",
                "source_mae": source_row["mae"],
                "driving_mae": driving_row["mae"],
                "group": "A_w2t",
            }
        )
    return pairs, missing


def build_b_s2w_132(existing_b_csv, exercise_filter):
    pairs = []
    with open(existing_b_csv, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            exercise = row.get("exercise", "")
            if exercise_filter and exercise != exercise_filter:
                continue
            if row.get("source_motion") != "Stationary":
                continue
            if row.get("driving_motion") != "Walking":
                continue
            row = dict(row)
            row["group"] = "B_s2w_132"
            pairs.append(row)
    return pairs


def validate_manifest(name, rows, mmpd_root, expected_count):
    errors = []
    if not rows:
        errors.append(f"{name}: no rows")
        return errors

    seen = set()
    dup_count = 0
    missing_paths = 0
    subject_mismatch = 0
    exercise_counter = Counter()

    for idx, row in enumerate(rows, start=2):
        key = (
            row.get("source_path", ""),
            row.get("driving_path", ""),
            str(row.get("subject", "")),
            row.get("light", ""),
            row.get("exercise", ""),
            row.get("source_motion", ""),
            row.get("driving_motion", ""),
        )
        if key in seen:
            dup_count += 1
        else:
            seen.add(key)

        source_path = row.get("source_path", "")
        driving_path = row.get("driving_path", "")
        source_abs = Path(source_path) if Path(source_path).is_absolute() else (mmpd_root / source_path)
        driving_abs = Path(driving_path) if Path(driving_path).is_absolute() else (mmpd_root / driving_path)
        if not source_abs.exists() or not driving_abs.exists():
            missing_paths += 1

        row_subject = int(row["subject"])
        source_subject = parse_subject_from_relpath(source_path)
        driving_subject = parse_subject_from_relpath(driving_path)
        if source_subject != row_subject or driving_subject != row_subject:
            subject_mismatch += 1

        exercise_counter[row.get("exercise", "")] += 1

    if dup_count:
        errors.append(f"{name}: duplicate rows={dup_count}")
    if missing_paths:
        errors.append(f"{name}: rows with missing source/driving path={missing_paths}")
    if subject_mismatch:
        errors.append(f"{name}: rows with subject/path mismatch={subject_mismatch}")
    if expected_count is not None and len(rows) != expected_count:
        errors.append(f"{name}: row count={len(rows)} expected={expected_count}")

    print(f"{name}: rows={len(rows)} exercises={dict(exercise_counter)}")
    return errors


def write_manifest(path, rows):
    fieldnames = [
        "source_path",
        "driving_path",
        "subject",
        "light",
        "exercise",
        "source_motion",
        "driving_motion",
        "source_mae",
        "driving_mae",
        "group",
    ]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmpd-root", default="datasets/MMPD")
    parser.add_argument(
        "--per-subject-condition",
        default="manifests/mmpd_cv/per_subject_condition.csv",
    )
    parser.add_argument(
        "--existing-b-csv",
        default="manifests/mmpd_cv/mmpd_pairs_B.csv",
    )
    parser.add_argument("--out-dir", default="manifests/mmpd_cv")
    parser.add_argument("--exercise", default="Normal")
    parser.add_argument("--expected-a", type=int, default=132)
    parser.add_argument("--expected-b", type=int, default=132)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    mmpd_root = Path(args.mmpd_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_map = load_mmpd_map(mmpd_root)
    cond_rows = load_conditions(args.per_subject_condition, args.exercise)
    a_rows, missing_a = build_a_w2t(cond_rows, file_map)
    b_rows = build_b_s2w_132(args.existing_b_csv, args.exercise)

    out_a = out_dir / "mmpd_pairs_A_w2t.csv"
    out_b = out_dir / "mmpd_pairs_B_s2w_132.csv"
    write_manifest(out_a, a_rows)
    write_manifest(out_b, b_rows)

    errors = []
    if missing_a:
        errors.append(f"A_w2t: missing mapped source/driving pairs={missing_a}")
    errors.extend(validate_manifest("A_w2t", a_rows, mmpd_root, args.expected_a))
    errors.extend(validate_manifest("B_s2w_132", b_rows, mmpd_root, args.expected_b))

    print(f"Wrote: {out_a}")
    print(f"Wrote: {out_b}")
    if errors:
        print("Manifest validation errors:")
        for err in errors:
            print(f"  - {err}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("Manifest validation passed.")


if __name__ == "__main__":
    main()
