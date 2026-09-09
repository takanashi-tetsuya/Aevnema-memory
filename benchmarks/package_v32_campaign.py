"""Create the v3.2 shareable archive while excluding answer-evidence logs."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import zipfile


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _shareable_files(campaign: Path) -> list[Path]:
    """Select reports and compact receipts, never raw answer-boundary payloads."""

    files: set[Path] = set()
    for directory in ("reports", "corpus", "source_gold", "experiments"):
        root = campaign / directory
        if not root.is_dir():
            continue
        files.update(root.rglob("*.md"))
    for name in ("campaign_state.json", "E32-00_baseline.json", "E32-01_n12c_stage_reconstruction.json"):
        path = campaign / name
        if path.is_file():
            files.add(path)
    for path in (
        campaign / "work_package_status.json",
        campaign / "reports" / "rollback_check.json",
    ):
        if path.is_file():
            files.add(path)
    for path in (
        campaign / "tests" / "E32-00_local_regression.json",
        campaign / "experiments" / "E32-04d_source_validated_cache_control" / "source_validated_cache_control.json",
        campaign / "experiments" / "E32-04d_source_validated_cache_control" / "paired_latency_results.json",
    ):
        if path.is_file():
            files.add(path)
    return sorted(
        path
        for path in files
        if ".full_local." not in path.name
        and ".answer-evidence." not in path.name
        and "logs" not in path.parts
        and path.suffix != ".sqlite"
    )


def package(*, campaign: Path) -> dict[str, object]:
    campaign = campaign.resolve()
    files = _shareable_files(campaign)
    local_only = sorted(
        path
        for path in campaign.rglob("*")
        if path.is_file()
        and (
            ".full_local." in path.name
            or ".answer-evidence." in path.name
            or "logs" in path.parts
            or path.suffix == ".sqlite"
        )
    )
    included = [
        {
            "path": path.relative_to(campaign).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in files
    ]
    withheld = [
        {
            "path": path.relative_to(campaign).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha(path),
        }
        for path in local_only
    ]
    manifest = {
        "schema": "aevnema.v3_2.redacted_delivery_manifest.v1",
        "campaign": str(campaign),
        "included": included,
        "withheld_local_only": withheld,
        "exclusion_policy": [
            "Exclude answer-evidence checkpoints and ordinary logs by default.",
            "Exclude *.full_local.* raw trails, SQLite snapshots and work directories from the shareable archive.",
            "Retain compact reports/receipts and hashes so locally authorized raw artifacts can be located without being silently shared.",
        ],
    }
    manifest_path = campaign / "ARTIFACT_MANIFEST.redacted.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    archive = campaign / "deliverable.redacted.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(campaign).as_posix())
        bundle.write(manifest_path, manifest_path.name)
    checksum_paths = [*files, manifest_path, archive, *local_only]
    checksum_path = campaign / "artifacts.sha256"
    checksum_path.write_text(
        "\n".join(
            f"{_sha(path)}  {path.relative_to(campaign).as_posix()}"
            for path in checksum_paths
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "archive": str(archive),
        "archive_sha256": _sha(archive),
        "included_count": len(files),
        "withheld_local_only_count": len(local_only),
        "manifest": str(manifest_path),
        "checksums": str(checksum_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(package(campaign=args.campaign_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
