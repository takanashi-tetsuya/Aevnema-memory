"""Validate Source-anchored recall gold without models, imports, or databases.

This evaluator-only tool intentionally reads only JSON artifacts and frozen
source files.  It never imports the application, opens a database, invokes a
model, or writes a knowledge base.  Its report contains identifiers, paths,
hashes, and counts only; it deliberately never includes source text or an
evidence span's contents.

``source_root`` is a local, quiescent snapshot input, not a live mutable
corpus directory.  The validator performs before/after path and metadata
checks and fails closed when it notices a change, but no pathname-only Python
implementation can prove safety against an adversary continuously replacing
Windows junctions or parent directories.  Formal use therefore requires a
controlled immutable snapshot (for example, a read-only copied corpus).

Approved Source gold is bound to a frozen-source manifest with this minimal
contract::

    {
      "schema": "aevnema.source-gold.frozen-source-manifest.v1",
      "sources": [
        {
          "source_key": "main/example.json",
          "source_file_sha256": "<64 lowercase hexadecimal characters>"
        }
      ]
    }

``gold.source_manifest_sha256`` must be the SHA-256 of that manifest's raw
bytes.  Every approved evidence atom must name a source from the frozen scope
and carry ``source_file_sha256``, a JSON Pointer ``record_locator``, integral
``span_start``/``span_end``, and ``raw_span_sha256``.  Span offsets are Python
Unicode code-point offsets into the JSON Pointer target string.  The pointer
may be written directly as a string, as ``{"json_pointer": "..."}``, or for
Blue Archive's native content records as
``{"kind":"blue_archive.content_record.v1","pointer":"/content/17/TextCn","record_index":17}``.

An approved artifact must also use ``schema_version``
``aevnema.source-gold.v1`` and explicitly set ``usable_for_scoring: true``.
At the gold, family, claim-group, and evidence-atom levels it must carry a
``review_status`` of ``accepted``, ``accepted_for_scoring``, ``approved``, or
``approved_for_scoring``.  These are deliberate scoring gates rather than
optional annotation fields.

Draft inputs are intentionally diagnosable.  Their defects are returned under
``diagnostics`` and their report is never ``scoring_eligible``.  This permits a
review worklist to be inspected without accidentally promoting it to an
experiment-ready gold set.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import ctypes
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from stat import S_ISDIR, S_ISREG
from tempfile import NamedTemporaryFile
from typing import Any


FROZEN_SOURCE_MANIFEST_SCHEMA = "aevnema.source-gold.frozen-source-manifest.v1"
SOURCE_GOLD_VALIDATION_SCHEMA = "aevnema.source-gold.validation-report.v1"
APPROVED_GOLD_SCHEMA = "aevnema.source-gold.v1"
DRAFT_GOLD_SCHEMA = "aevnema.source-gold.draft.v1"
APPROVED_SPLIT_SCHEMA = "aevnema.gold-split.v1"
DRAFT_SPLIT_SCHEMA = "aevnema.gold-split.draft.v1"

_APPROVED_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_scoring",
        "approved",
        "approved_for_scoring",
    }
)
_DRAFT_STATUSES = frozenset(
    {
        "draft",
        "draft_pending_source_span_review",
        "pending_source_span_review",
    }
)
_APPROVED_REVIEW_STATUSES = frozenset(
    {
        "accepted",
        "accepted_for_scoring",
        "approved",
        "approved_for_scoring",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_ARRAY_INDEX_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
_MAX_ARRAY_INDEX_DIGITS = 19
_MAX_JSON_NESTING = 512

_PathIdentity = tuple[int, int, int, int]
_PathSnapshot = tuple[Path, _PathIdentity]


class SourceGoldValidationInputError(ValueError):
    """Raised when a formal JSON input is unsafe, unreadable, or not JSON."""


def _is_mapping(value: object) -> bool:
    return isinstance(value, Mapping)


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _normalise_sha256(value: object) -> str | None:
    """Return a digest without its optional conventional ``sha256:`` prefix."""

    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if candidate.lower().startswith("sha256:"):
        candidate = candidate[7:]
    if not _SHA256_RE.fullmatch(candidate):
        return None
    return candidate.lower()


def _is_approved_status(value: object) -> bool:
    return isinstance(value, str) and value in _APPROVED_STATUSES


def _is_draft_status(value: object) -> bool:
    return isinstance(value, str) and value in _DRAFT_STATUSES


def _is_approved_review_status(value: object) -> bool:
    return isinstance(value, str) and value in _APPROVED_REVIEW_STATUSES


def _status_category(value: object) -> str:
    """Classify an untrusted status without reflecting it into a report."""

    if _is_approved_status(value):
        return "approved"
    if _is_draft_status(value):
        return "draft"
    if value is None:
        return "missing"
    return "other"


def _strict_json_loads(raw: bytes) -> object:
    """Reject excessive nesting, duplicate keys, and non-finite constants."""

    # CPython's JSON recursion ceiling differs between platform builds. Check
    # container nesting before decoding instead of relying on RecursionError.
    depth, in_string, escaped = 0, False, False
    for byte in raw:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 92:  # backslash
                escaped = True
            elif byte == 34:  # double quote
                in_string = False
        elif byte == 34:
            in_string = True
        elif byte in (91, 123):  # array or object opening
            depth += 1
            if depth > _MAX_JSON_NESTING:
                raise ValueError("JSON container nesting exceeds supported limit")
        elif byte in (93, 125):
            depth -= 1

    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    def reject_nonfinite(_value: str) -> object:
        raise ValueError("non-finite JSON number")

    def reject_nonfinite_float(value: str) -> float:
        """Reject finite-looking JSON literals that overflow to infinity."""

        parsed = float(value)
        if not isfinite(parsed):
            raise ValueError("non-finite JSON number")
        return parsed

    return json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_nonfinite,
        parse_float=reject_nonfinite_float,
    )


def _issue(code: str, message: str, *, path: str, scope: str) -> dict[str, str]:
    """Create a text-safe report item.

    Callers must keep ``message`` generic.  In particular, it must not contain
    a source value, JSON leaf text, or a rejected pointer component.
    """

    return {"scope": scope, "code": code, "path": path, "message": message}


def _append_problem(
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    item: dict[str, str],
    *,
    draft: bool,
) -> None:
    """Keep draft review useful without treating it as scoring-ready."""

    if draft:
        diagnostics.append(item)
    else:
        issues.append(item)


def _safe_source_key(value: object) -> str | None:
    """Validate a corpus-relative POSIX source key without resolving it yet."""

    if not _is_nonempty_string(value):
        return None
    key = str(value)
    if "\\" in key:
        return None
    candidate = PurePosixPath(key)
    windows_candidate = PureWindowsPath(key)
    if (
        candidate.is_absolute()
        or windows_candidate.is_absolute()
        or windows_candidate.drive
        or windows_candidate.root
        or not candidate.parts
        or key != candidate.as_posix()
        or key != key.casefold()
    ):
        return None
    if any(
        part in {"", ".", ".."}
        or ":" in part
        or part.rstrip(" .") != part
        or PureWindowsPath(part).is_reserved()
        for part in candidate.parts
    ):
        return None
    return candidate.as_posix()


def _path_snapshot(path: Path) -> _PathSnapshot | None:
    """Capture a resolved path plus stable metadata for change detection."""

    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
    except (OSError, RuntimeError):
        return None
    return (
        resolved,
        (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
        ),
    )


def _snapshot_is_directory(snapshot: _PathSnapshot | None) -> bool:
    """Check a captured root safely if it still exists at this instant."""

    if snapshot is None:
        return False
    try:
        return snapshot[0].is_dir()
    except OSError:
        return False


def _is_link_or_junction(path: Path) -> bool:
    """Treat Windows junctions like symlinks when the runtime exposes them."""

    try:
        is_junction = getattr(path, "is_junction", lambda: False)
        return path.is_symlink() or bool(is_junction())
    except OSError:
        return True


def _is_windows_unc_path(path: str | Path) -> bool:
    """Recognize UNC syntax without attempting to contact a network share."""

    text = str(path)
    if text.startswith(("\\\\", "//")):
        return True
    try:
        return PureWindowsPath(text).drive.startswith("\\\\")
    except (TypeError, ValueError):
        return True


def _windows_drive_root_without_resolving(path: str | Path) -> str | None:
    """Return a Windows drive root without following a path or link.

    A drive-relative spelling such as ``C:artifact.json`` is deliberately not
    accepted.  It depends on mutable per-drive working-directory state and is
    therefore not a direct, reproducible artifact location.
    """

    try:
        candidate = PureWindowsPath(str(path))
        if candidate.drive:
            if candidate.drive.startswith("\\\\") or candidate.root != "\\":
                return None
            return f"{candidate.drive}\\"
        cwd = PureWindowsPath(os.getcwd())
        if (
            not cwd.drive
            or cwd.drive.startswith("\\\\")
            or cwd.root != "\\"
        ):
            return None
        return f"{cwd.drive}\\"
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _windows_get_drive_type(path: str | Path) -> int | None:
    """Call ``GetDriveTypeW`` without resolving an untrusted input path.

    ``None`` means localness could not be established.  It must never be
    treated as a local result by a formal Source-gold run.
    """

    if os.name != "nt":
        return None
    drive_root = _windows_drive_root_without_resolving(path)
    if drive_root is None:
        return None
    try:
        get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
        return int(get_drive_type(drive_root))
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None


def _is_windows_remote_drive(path: str | Path) -> bool:
    """Compatibility predicate that never treats an unverifiable drive as local."""

    return _windows_get_drive_type(path) == 4  # DRIVE_REMOTE


def _windows_locality_rejection_code(
    path: str | Path,
    *,
    prefix: str,
) -> str | None:
    """Fail closed unless Windows positively classifies the drive as local."""

    if os.name != "nt":
        return None
    if _is_windows_unc_path(path):
        return f"{prefix}_unc_not_allowed"
    drive_type = _windows_get_drive_type(path)
    if drive_type == 4:  # DRIVE_REMOTE
        return f"{prefix}_remote_drive_not_allowed"
    # DRIVE_REMOVABLE, DRIVE_FIXED, DRIVE_CDROM, and DRIVE_RAMDISK are local
    # drive classes.  DRIVE_UNKNOWN (0), DRIVE_NO_ROOT_DIR (1), missing API
    # support, and every other value do not prove a direct local input.
    if drive_type not in {2, 3, 5, 6}:
        return f"{prefix}_locality_unverifiable"
    return None


def _absolute_path_without_resolving(path: Path) -> Path | None:
    """Make a lexical absolute path without following links or junctions."""

    try:
        return Path(os.path.abspath(os.fspath(path)))
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _has_link_or_junction_component(path: Path) -> bool:
    """Reject a final path or any parent that is a link/junction.

    This intentionally uses only lexical absolute-path construction and
    ``lstat``-style link checks.  It runs before ``Path.resolve`` and before a
    JSON artifact is opened, so a normal unsafe spelling never causes a remote
    share or link target to be traversed.
    """

    direct_path = _absolute_path_without_resolving(path)
    if direct_path is None:
        return True
    try:
        # Check root-to-leaf.  Checking the leaf first can make Windows lstat
        # traverse a parent junction before the junction itself is rejected.
        components = tuple(reversed((direct_path, *direct_path.parents)))
    except (OSError, RuntimeError):
        return True
    return any(_is_link_or_junction(component) for component in components)


def _direct_local_path_preflight(
    path: str | Path,
    *,
    prefix: str,
    expect_directory: bool,
    allow_missing_leaf: bool = False,
) -> tuple[Path | None, str | None]:
    """Verify a direct local file/directory before resolving or reading it.

    The policy is intentionally stricter than ordinary file loading: UNC
    syntax, mapped remote drives, direct or parent symlinks/junctions, missing
    metadata, and unclassified Windows drives are all rejected.  A later
    snapshot/read check still detects ordinary changes after this preflight;
    formal callers must provide a quiescent immutable snapshot to close the
    remaining concurrent-filesystem race.
    """

    if _is_windows_unc_path(path):
        return None, f"{prefix}_unc_not_allowed"
    # Preserve the raw-spelling check for drive-relative Windows inputs, then
    # check the fixed absolute spelling below as well.  This avoids a CWD
    # change rebinding a relative input between drive classification and use.
    windows_rejection = _windows_locality_rejection_code(path, prefix=prefix)
    if windows_rejection is not None:
        return None, windows_rejection
    try:
        requested = Path(path)
    except (TypeError, ValueError):
        return None, f"{prefix}_unavailable"
    direct_path = _absolute_path_without_resolving(requested)
    if direct_path is None:
        return None, f"{prefix}_unavailable"
    if _is_windows_unc_path(direct_path):
        return None, f"{prefix}_unc_not_allowed"
    windows_rejection = _windows_locality_rejection_code(
        direct_path,
        prefix=prefix,
    )
    if windows_rejection is not None:
        return None, windows_rejection
    if _has_link_or_junction_component(direct_path):
        return None, f"{prefix}_link_or_junction_not_allowed"
    try:
        metadata = direct_path.lstat()
    except FileNotFoundError:
        if allow_missing_leaf:
            return direct_path, None
        return None, f"{prefix}_unavailable"
    except (OSError, RuntimeError):
        return None, f"{prefix}_unavailable"
    if expect_directory:
        if S_ISDIR(metadata.st_mode):
            return direct_path, None
        return None, f"{prefix}_unavailable"
    if S_ISREG(metadata.st_mode):
        return direct_path, None
    return None, f"{prefix}_unavailable"


def _direct_local_path_rejection_code(
    path: str | Path,
    *,
    prefix: str,
    expect_directory: bool,
) -> str | None:
    """Return only the rejection part of :func:`_direct_local_path_preflight`."""

    _direct_path, rejection = _direct_local_path_preflight(
        path,
        prefix=prefix,
        expect_directory=expect_directory,
    )
    return rejection


def _source_root_rejection_code(source_root: str | Path) -> str | None:
    """Return a safe reason when ``source_root`` is not a direct local tree."""

    return _direct_local_path_rejection_code(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )


def _artifact_input_rejection_code(
    path: str | Path,
    *,
    label: str,
) -> str | None:
    """Return a non-sensitive rejection code for one formal JSON input."""

    return _direct_local_path_rejection_code(
        path,
        prefix=f"{label}_input",
        expect_directory=False,
    )


def _require_direct_local_artifact_inputs(
    *,
    gold_path: str | Path,
    split_path: str | Path,
    frozen_source_manifest_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Return fixed direct-local artifact paths before any read or resolve."""

    verified_paths: list[Path] = []
    for label, artifact_path in (
        ("gold", gold_path),
        ("split", split_path),
        ("frozen_source_manifest", frozen_source_manifest_path),
    ):
        direct_path, rejection = _direct_local_path_preflight(
            artifact_path,
            prefix=f"{label}_input",
            expect_directory=False,
        )
        if rejection is not None or direct_path is None:
            raise SourceGoldValidationInputError(
                "requested JSON artifact is not a permitted direct local file"
            )
        verified_paths.append(direct_path)
    return verified_paths[0], verified_paths[1], verified_paths[2]


def _safe_source_path(source_root: Path, source_key: str) -> Path | None:
    """Resolve a source key while refusing traversal and source-root escapes."""

    try:
        root = source_root.resolve(strict=True)
        raw_target = root
        for part in PurePosixPath(source_key).parts:
            raw_target = raw_target / part
            if _is_link_or_junction(raw_target):
                return None
        target = raw_target.resolve(strict=True)
        target.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    if not target.is_file() or target.is_symlink():
        return None
    return target


def _parse_array_index(token: str) -> int | None:
    """Parse a practical JSON-array index without unbounded integer work."""

    if not _ARRAY_INDEX_RE.fullmatch(token) or len(token) > _MAX_ARRAY_INDEX_DIGITS:
        return None
    try:
        return int(token)
    except ValueError:
        return None


def _source_file_identity(path: Path) -> tuple[object, ...] | None:
    """Return an internal physical-file identity without placing it in reports."""

    snapshot = _path_snapshot(path)
    if snapshot is None:
        return None
    resolved, metadata = snapshot
    # A pathname fallback cannot prove source-disjointness against Windows
    # aliases or hard links.  Filesystems without a stable file identifier are
    # therefore unsupported for formal Source-gold validation.
    if metadata[1] == 0:
        return None
    return ("inode", metadata[0], metadata[1])


def _validate_frozen_source_identities(
    frozen_sources: Mapping[str, str],
    *,
    source_root: Path,
    source_root_snapshot: _PathSnapshot | None,
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> dict[str, tuple[object, ...]]:
    """Bind each declared key to one physical file and reject aliases.

    A holdout split cannot safely treat two spellings of the same Windows file,
    hard link, or other physical alias as independent sources.  The returned
    identities are internal-only and never emitted in the report.
    """

    identities: dict[str, tuple[object, ...]] = {}
    seen: dict[tuple[object, ...], str] = {}
    if source_root_snapshot is None or _path_snapshot(source_root) != source_root_snapshot:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "source_root_changed_during_validation",
                "source_root changed while the frozen source scope was being checked",
                path="source_root",
                scope="source_manifest",
            ),
            draft=draft,
        )
        return identities
    for source_key in frozen_sources:
        source_path = _safe_source_path(source_root, source_key)
        if source_path is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_unavailable",
                    "a frozen source binding cannot be resolved safely under source_root",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        identity = _source_file_identity(source_path)
        if identity is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_identity_unavailable",
                    "a frozen source binding has no stable local file identity",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        if _path_snapshot(source_root) != source_root_snapshot:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "source_root_changed_during_validation",
                    "source_root changed while the frozen source scope was being checked",
                    path="source_root",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            return identities
        if identity in seen:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_physical_alias",
                    "frozen source manifest maps more than one source_key to one physical file",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        seen[identity] = source_key
        identities[source_key] = identity
    return identities


def _validate_frozen_source_contents(
    frozen_sources: Mapping[str, str],
    *,
    source_root: Path,
    source_root_snapshot: _PathSnapshot | None,
    source_identities: Mapping[str, tuple[object, ...]],
    source_cache: dict[str, tuple[str | None, object | None, _PathSnapshot | None]],
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> None:
    """Verify every frozen source revision before any atom can use it.

    Evidence atoms usually cover only part of a frozen corpus.  A manifest is
    nevertheless a complete snapshot contract: an unreferenced entry that is
    missing, changed, or malformed must not leave an approved gold ready to
    score.  Each accepted read is cached with its snapshot so the final
    stability check covers the full manifest rather than only atom references.
    """

    if source_root_snapshot is None:
        return
    for source_key, expected_source_digest in frozen_sources.items():
        if _path_snapshot(source_root) != source_root_snapshot:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "source_root_changed_during_validation",
                    "source_root changed while frozen source contents were being checked",
                    path="source_root",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            return

        source_path = _safe_source_path(source_root, source_key)
        source_file_snapshot = (
            _path_snapshot(source_path) if source_path is not None else None
        )
        current_source_identity = (
            _source_file_identity(source_path) if source_path is not None else None
        )
        expected_source_identity = source_identities.get(source_key)
        if (
            source_path is None
            or source_file_snapshot is None
            or current_source_identity is None
        ):
            source_cache[source_key] = (None, None, None)
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_unavailable",
                    "a frozen source file cannot be read safely under source_root",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        if (
            expected_source_identity is not None
            and current_source_identity != expected_source_identity
        ):
            source_cache[source_key] = (None, None, None)
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_identity_changed_during_validation",
                    "a frozen source file changed physical identity before its contents were read",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue

        source_digest, source_document = _read_source_file(
            source_path,
            source_root=source_root,
            source_key=source_key,
            expected_root_snapshot=source_root_snapshot,
            expected_file_snapshot=source_file_snapshot,
        )
        source_cache[source_key] = (
            source_digest,
            source_document,
            source_file_snapshot,
        )
        if source_digest is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_unavailable",
                    "a frozen source file cannot be read safely under source_root",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        if source_digest != expected_source_digest:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_sha256_mismatch",
                    "local source file bytes do not match the frozen source manifest",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
        if source_document is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_json_unreadable",
                    "frozen source file is not valid UTF-8 JSON",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )


def _parse_json_pointer(locator: object) -> tuple[list[str] | None, int | None, str | None]:
    """Parse a JSON Pointer locator without returning untrusted text in errors."""

    value: object = locator
    expected_record_index: int | None = None
    if _is_mapping(locator):
        if set(locator) == {"json_pointer"}:
            value = locator.get("json_pointer")
        elif set(locator) == {"kind", "pointer", "record_index"}:
            if locator.get("kind") != "blue_archive.content_record.v1":
                return None, None, "record_locator kind is not recognized"
            value = locator.get("pointer")
            raw_record_index = locator.get("record_index")
            if isinstance(raw_record_index, bool) or not isinstance(raw_record_index, int):
                return None, None, "record_locator record_index must be an integer"
            if raw_record_index < 0:
                return None, None, "record_locator record_index must be non-negative"
            expected_record_index = raw_record_index
        else:
            return None, None, "record_locator object shape is not recognized"
    if not isinstance(value, str):
        return None, None, "record_locator must be a JSON Pointer string"
    if value == "":
        return None, None, "record_locator must identify a non-root source record"
    if not value.startswith("/"):
        return None, None, "record_locator must be a JSON Pointer"

    tokens: list[str] = []
    for raw_token in value[1:].split("/"):
        token: list[str] = []
        position = 0
        while position < len(raw_token):
            character = raw_token[position]
            if character != "~":
                token.append(character)
                position += 1
                continue
            if position + 1 >= len(raw_token) or raw_token[position + 1] not in {"0", "1"}:
                return None, None, "record_locator has an invalid JSON Pointer escape"
            token.append("~" if raw_token[position + 1] == "0" else "/")
            position += 2
        tokens.append("".join(token))
    if expected_record_index is not None:
        if (
            len(tokens) < 2
            or tokens[0] != "content"
            or _parse_array_index(tokens[1]) is None
            or _parse_array_index(tokens[1]) != expected_record_index
        ):
            return None, None, "record_locator record_index does not match its content pointer"
    return tokens, expected_record_index, None


def _resolve_json_pointer(document: object, tokens: Sequence[str]) -> tuple[object | None, str | None]:
    """Resolve a parsed JSON Pointer without revealing the rejected record."""

    current = document
    for token in tokens:
        if isinstance(current, Mapping):
            if token not in current:
                return None, "record_locator does not resolve in the source file"
            current = current[token]
            continue
        if isinstance(current, list):
            index = _parse_array_index(token)
            if index is None:
                return None, "record_locator does not select a source array record"
            if index >= len(current):
                return None, "record_locator does not resolve in the source file"
            current = current[index]
            continue
        return None, "record_locator traverses through a non-container value"
    return current, None


def _read_source_file(
    path: Path,
    *,
    source_root: Path,
    source_key: str,
    expected_root_snapshot: _PathSnapshot,
    expected_file_snapshot: _PathSnapshot,
) -> tuple[str | None, object | None]:
    """Read/hash/parse one source revision from the same bytes object.

    Hashing an earlier read and parsing a later read would permit a local
    time-of-check/time-of-use mismatch.  This helper deliberately derives both
    facts from one byte sequence.  Parsing failures do not expose source text.
    """

    try:
        raw = path.read_bytes()
    except OSError:
        return None, None
    current_root_snapshot = _path_snapshot(source_root)
    current_path = _safe_source_path(source_root, source_key)
    current_file_snapshot = (
        _path_snapshot(current_path) if current_path is not None else None
    )
    if (
        current_root_snapshot != expected_root_snapshot
        or current_file_snapshot != expected_file_snapshot
    ):
        return None, None
    digest = sha256(raw).hexdigest()
    try:
        return digest, _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError):
        return digest, None


def _read_json_bytes(path: Path, *, label: str) -> bytes:
    """Read a previously-approved local artifact from one stable byte sequence.

    The direct-local policy is deliberately repeated here.  The file API first
    checks *all* three artifact paths before any read; this second check catches
    an ordinary path replacement between that batch precheck and this specific
    open.  It is a detection boundary rather than a claim to defeat a hostile
    concurrent filesystem, which formal callers avoid by using immutable
    artifact inputs.
    """

    direct_path, rejection = _direct_local_path_preflight(
        path,
        prefix=f"{label}_input",
        expect_directory=False,
    )
    if rejection is not None or direct_path is None:
        raise SourceGoldValidationInputError(
            "requested JSON artifact is not a permitted direct local file"
        )
    snapshot = _path_snapshot(direct_path)
    if snapshot is None:
        raise SourceGoldValidationInputError("cannot read requested JSON artifact")
    try:
        raw = direct_path.read_bytes()
    except OSError as exc:
        raise SourceGoldValidationInputError("cannot read requested JSON artifact") from exc
    current_path, current_rejection = _direct_local_path_preflight(
        direct_path,
        prefix=f"{label}_input",
        expect_directory=False,
    )
    if (
        current_rejection is not None
        or current_path != direct_path
        or _path_snapshot(direct_path) != snapshot
    ):
        raise SourceGoldValidationInputError(
            "requested JSON artifact changed while it was being read"
        )
    return raw


def _parse_artifact_json(raw: bytes) -> object:
    """Strictly parse an artifact without reflecting its content into an error."""

    try:
        return _strict_json_loads(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError, RecursionError) as exc:
        raise SourceGoldValidationInputError("requested artifact is not valid UTF-8 JSON") from exc


def _parse_frozen_sources(
    frozen_source_manifest: object,
    *,
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> dict[str, str]:
    """Return frozen ``source_key -> file_sha256`` bindings, fail-closed."""

    if not _is_mapping(frozen_source_manifest):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_manifest_not_object",
                "frozen source manifest must be a JSON object",
                path="source_manifest",
                scope="source_manifest",
            ),
            draft=draft,
        )
        return {}
    if frozen_source_manifest.get("schema") != FROZEN_SOURCE_MANIFEST_SCHEMA:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_manifest_schema_invalid",
                "frozen source manifest schema is not recognized",
                path="source_manifest.schema",
                scope="source_manifest",
            ),
            draft=draft,
        )
    sources = frozen_source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_manifest_sources_missing",
                "frozen source manifest must contain at least one source binding",
                path="source_manifest.sources",
                scope="source_manifest",
            ),
            draft=draft,
        )
        return {}

    frozen: dict[str, str] = {}
    for index, entry in enumerate(sources):
        path = f"source_manifest.sources[{index}]"
        if not _is_mapping(entry):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_manifest_entry_invalid",
                    "each frozen source binding must be an object",
                    path=path,
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        source_key = _safe_source_key(entry.get("source_key"))
        if source_key is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_key_invalid",
                    "frozen source binding needs a safe corpus-relative source_key",
                    path=f"{path}.source_key",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        digest = _normalise_sha256(entry.get("source_file_sha256"))
        if digest is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_file_sha256_invalid",
                    "frozen source binding needs a SHA-256 source_file_sha256",
                    path=f"{path}.source_file_sha256",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        if source_key in frozen:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_key_duplicate",
                    "frozen source manifest must not repeat a source_key",
                    path=f"{path}.source_key",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            continue
        frozen[source_key] = digest
    return frozen


def _validate_atom(
    atom: object,
    *,
    path: str,
    source_root: Path,
    source_root_snapshot: _PathSnapshot | None,
    frozen_sources: Mapping[str, str],
    source_identities: Mapping[str, tuple[object, ...]],
    source_cache: dict[str, tuple[str | None, object | None, _PathSnapshot | None]],
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> tuple[str, tuple[object, ...] | None] | None:
    """Validate one atom and return its source key plus physical identity."""

    if not _is_mapping(atom):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_not_object",
                "each evidence atom must be a JSON object",
                path=path,
                scope="atom",
            ),
            draft=draft,
        )
        return None

    source_key = _safe_source_key(atom.get("source_key"))
    if source_key is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_source_key_invalid",
                "approved evidence atom needs a safe corpus-relative source_key",
                path=f"{path}.source_key",
                scope="atom",
            ),
            draft=draft,
        )
        return None
    if source_key not in frozen_sources:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_source_not_in_frozen_scope",
                "evidence atom source_key is absent from the frozen source scope",
                path=f"{path}.source_key",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)

    expected_source_digest = frozen_sources[source_key]
    actual_source_digest: str | None
    source_document: object | None
    if source_key not in source_cache:
        source_path = _safe_source_path(source_root, source_key)
        source_file_snapshot = (
            _path_snapshot(source_path) if source_path is not None else None
        )
        current_source_identity = (
            _source_file_identity(source_path) if source_path is not None else None
        )
        if (
            source_root_snapshot is None
            or source_file_snapshot is None
            or source_path is None
            or current_source_identity != source_identities.get(source_key)
        ):
            source_cache[source_key] = (None, None, None)
            if source_path is not None and current_source_identity != source_identities.get(source_key):
                _append_problem(
                    issues,
                    diagnostics,
                    _issue(
                        "frozen_source_identity_changed_during_validation",
                        "a frozen source file changed physical identity before evidence was read",
                        path=f"{path}.source_key",
                        scope="atom",
                    ),
                    draft=draft,
                )
        else:
            source_digest, source_document = _read_source_file(
                source_path,
                source_root=source_root,
                source_key=source_key,
                expected_root_snapshot=source_root_snapshot,
                expected_file_snapshot=source_file_snapshot,
            )
            source_cache[source_key] = (
                source_digest,
                source_document,
                source_file_snapshot,
            )
    actual_source_digest, source_document, _source_file_snapshot = source_cache[source_key]
    if actual_source_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_file_unavailable",
                "a frozen source file cannot be read safely under source_root",
                path=f"{path}.source_key",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    if actual_source_digest != expected_source_digest:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_file_sha256_mismatch",
                "local source file bytes do not match the frozen source manifest",
                path=f"{path}.source_key",
                scope="atom",
            ),
            draft=draft,
        )

    atom_source_digest = _normalise_sha256(atom.get("source_file_sha256"))
    if atom_source_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_source_file_sha256_missing",
                "approved evidence atom needs a SHA-256 source_file_sha256",
                path=f"{path}.source_file_sha256",
                scope="atom",
            ),
            draft=draft,
        )
    elif atom_source_digest != expected_source_digest or atom_source_digest != actual_source_digest:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_source_file_sha256_mismatch",
                "atom source_file_sha256 does not bind the frozen local source file",
                path=f"{path}.source_file_sha256",
                scope="atom",
            ),
            draft=draft,
        )

    tokens, _record_index, locator_error = _parse_json_pointer(atom.get("record_locator"))
    if tokens is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_record_locator_invalid",
                locator_error or "record_locator is invalid",
                path=f"{path}.record_locator",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    if source_document is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_json_unreadable",
                "frozen source file is not valid UTF-8 JSON",
                path=f"{path}.record_locator",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    record, record_error = _resolve_json_pointer(source_document, tokens)
    if record_error is not None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_record_locator_unresolved",
                record_error,
                path=f"{path}.record_locator",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    if not isinstance(record, str):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_record_not_text",
                "record_locator must resolve to a JSON string before span validation",
                path=f"{path}.record_locator",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)

    span_start = atom.get("span_start")
    span_end = atom.get("span_end")
    if (
        isinstance(span_start, bool)
        or isinstance(span_end, bool)
        or not isinstance(span_start, int)
        or not isinstance(span_end, int)
    ):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_span_not_integer",
                "approved evidence atom needs integer span_start and span_end",
                path=path,
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    if span_start < 0 or span_end <= span_start or span_end > len(record):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_span_out_of_bounds",
                "evidence span must be non-empty and within its located source record",
                path=path,
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)

    raw_span_digest = _normalise_sha256(atom.get("raw_span_sha256"))
    if raw_span_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_raw_span_sha256_missing",
                "approved evidence atom needs a SHA-256 raw_span_sha256",
                path=f"{path}.raw_span_sha256",
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    try:
        raw_span_bytes = record[span_start:span_end].encode("utf-8")
    except UnicodeEncodeError:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_span_not_utf8",
                "located source span is not valid UTF-8 text",
                path=path,
                scope="atom",
            ),
            draft=draft,
        )
        return source_key, source_identities.get(source_key)
    actual_span_digest = sha256(raw_span_bytes).hexdigest()
    if raw_span_digest != actual_span_digest:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "atom_raw_span_sha256_mismatch",
                "raw_span_sha256 does not match the located local source span",
                path=f"{path}.raw_span_sha256",
                scope="atom",
            ),
            draft=draft,
        )
    return source_key, source_identities.get(source_key)


def _validate_gold_structure(
    gold: object,
    *,
    source_root: Path,
    source_root_snapshot: _PathSnapshot | None,
    frozen_sources: Mapping[str, str],
    source_identities: Mapping[str, tuple[object, ...]],
    source_cache: dict[str, tuple[str | None, object | None, _PathSnapshot | None]],
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> tuple[dict[str, set[str]], dict[str, set[tuple[object, ...]]], list[str], int, int]:
    """Validate families and return declared/physical source sets plus counts."""

    family_sources: dict[str, set[str]] = {}
    family_source_identities: dict[str, set[tuple[object, ...]]] = {}
    family_ids: list[str] = []
    group_count = 0
    atom_count = 0
    if not _is_mapping(gold):
        return family_sources, family_source_identities, family_ids, group_count, atom_count
    families = gold.get("families")
    if not isinstance(families, list) or not families:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_missing_families",
                "gold must contain at least one family",
                path="gold.families",
                scope="gold",
            ),
            draft=draft,
        )
        return family_sources, family_source_identities, family_ids, group_count, atom_count

    for family_index, family in enumerate(families):
        family_path = f"gold.families[{family_index}]"
        if not _is_mapping(family):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "gold_family_not_object",
                    "each gold family must be a JSON object",
                    path=family_path,
                    scope="gold",
                ),
                draft=draft,
            )
            continue
        family_id = family.get("family_id")
        if not _is_nonempty_string(family_id):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "gold_family_id_missing",
                    "each gold family needs a non-empty family_id",
                    path=f"{family_path}.family_id",
                    scope="gold",
                ),
                draft=draft,
            )
            continue
        family_id = str(family_id)
        if family_id in family_sources:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "gold_family_id_duplicate",
                    "gold family_id values must be unique",
                    path=f"{family_path}.family_id",
                    scope="gold",
                ),
                draft=draft,
            )
            continue
        family_ids.append(family_id)
        family_sources[family_id] = set()
        family_source_identities[family_id] = set()
        if family.get("usable_for_scoring") is not True:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "gold_family_not_usable_for_scoring",
                    "approved gold family must explicitly set usable_for_scoring to true",
                    path=f"{family_path}.usable_for_scoring",
                    scope="gold",
                ),
                draft=draft,
            )
        if not draft and not _is_approved_review_status(family.get("review_status")):
            issues.append(
                _issue(
                    "gold_family_review_status_not_approved",
                    "approved gold family needs an explicit approved review_status",
                    path=f"{family_path}.review_status",
                    scope="gold",
                )
            )
        groups = family.get("claim_groups")
        if not isinstance(groups, list) or not groups:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "gold_claim_groups_missing",
                    "each gold family needs at least one claim group",
                    path=f"{family_path}.claim_groups",
                    scope="gold",
                ),
                draft=draft,
            )
            continue
        seen_group_ids: set[str] = set()
        for group_index, group in enumerate(groups):
            group_count += 1
            group_path = f"{family_path}.claim_groups[{group_index}]"
            if not _is_mapping(group):
                _append_problem(
                    issues,
                    diagnostics,
                    _issue(
                        "gold_claim_group_not_object",
                        "each claim group must be a JSON object",
                        path=group_path,
                        scope="gold",
                    ),
                    draft=draft,
                )
                continue
            group_id = group.get("claim_group_id")
            if not _is_nonempty_string(group_id) or str(group_id) in seen_group_ids:
                _append_problem(
                    issues,
                    diagnostics,
                    _issue(
                        "gold_claim_group_id_invalid",
                        "claim_group_id must be present and unique within its family",
                        path=f"{group_path}.claim_group_id",
                        scope="gold",
                    ),
                    draft=draft,
                )
            else:
                seen_group_ids.add(str(group_id))
            if group.get("usable_for_scoring") is not True:
                _append_problem(
                    issues,
                    diagnostics,
                    _issue(
                        "gold_claim_group_not_usable_for_scoring",
                        "approved claim group must explicitly set usable_for_scoring to true",
                        path=f"{group_path}.usable_for_scoring",
                        scope="gold",
                    ),
                    draft=draft,
                )
            if not draft and not _is_approved_review_status(group.get("review_status")):
                issues.append(
                    _issue(
                        "gold_claim_group_review_status_not_approved",
                        "approved claim group needs an explicit approved review_status",
                        path=f"{group_path}.review_status",
                        scope="gold",
                    )
                )
            atoms = group.get("evidence_atoms")
            if not isinstance(atoms, list) or not atoms:
                _append_problem(
                    issues,
                    diagnostics,
                    _issue(
                        "gold_evidence_atoms_missing",
                        "each claim group needs at least one evidence atom",
                        path=f"{group_path}.evidence_atoms",
                        scope="gold",
                    ),
                    draft=draft,
                )
                continue
            for atom_index, atom in enumerate(atoms):
                atom_count += 1
                atom_path = f"{group_path}.evidence_atoms[{atom_index}]"
                if _is_mapping(atom) and atom.get("usable_for_scoring") is not True:
                    _append_problem(
                        issues,
                        diagnostics,
                        _issue(
                            "gold_atom_not_usable_for_scoring",
                            "approved evidence atom must explicitly set usable_for_scoring to true",
                            path=f"{atom_path}.usable_for_scoring",
                            scope="gold",
                        ),
                        draft=draft,
                    )
                if _is_mapping(atom) and not draft and not _is_approved_review_status(
                    atom.get("review_status")
                ):
                    issues.append(
                        _issue(
                            "gold_atom_review_status_not_approved",
                            "approved evidence atom needs an explicit approved review_status",
                            path=f"{atom_path}.review_status",
                            scope="gold",
                        )
                    )
                source_binding = _validate_atom(
                    atom,
                    path=atom_path,
                    source_root=source_root,
                    source_root_snapshot=source_root_snapshot,
                    frozen_sources=frozen_sources,
                    source_identities=source_identities,
                    source_cache=source_cache,
                    issues=issues,
                    diagnostics=diagnostics,
                    draft=draft,
                )
                if source_binding is not None:
                    source_key, source_identity = source_binding
                    family_sources[family_id].add(source_key)
                    if source_identity is not None:
                        family_source_identities[family_id].add(source_identity)
    return family_sources, family_source_identities, family_ids, group_count, atom_count


def _validate_source_cache_stability(
    source_cache: Mapping[str, tuple[str | None, object | None, _PathSnapshot | None]],
    *,
    source_root: Path,
    source_root_snapshot: _PathSnapshot | None,
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> None:
    """Best-effort final check that the quiescent source snapshot stayed stable."""

    if source_root_snapshot is None:
        return
    if _path_snapshot(source_root) != source_root_snapshot:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "source_root_changed_during_validation",
                "source_root changed while local source evidence was being checked",
                path="source_root",
                scope="source_manifest",
            ),
            draft=draft,
        )
        return
    for source_key, (_digest, _document, expected_file_snapshot) in source_cache.items():
        source_path = _safe_source_path(source_root, source_key)
        current_file_snapshot = (
            _path_snapshot(source_path) if source_path is not None else None
        )
        if expected_file_snapshot is None or current_file_snapshot != expected_file_snapshot:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "frozen_source_changed_during_validation",
                    "a frozen source file changed while local evidence was being checked",
                    path="source_manifest.sources",
                    scope="source_manifest",
                ),
                draft=draft,
            )
            return


def _safe_source_key_list(value: object) -> set[str] | None:
    if not isinstance(value, list):
        return None
    keys: set[str] = set()
    for raw_key in value:
        key = _safe_source_key(raw_key)
        if key is None or key in keys:
            return None
        keys.add(key)
    return keys


def _validate_split(
    split: object,
    *,
    family_sources: Mapping[str, set[str]],
    family_source_identities: Mapping[str, set[tuple[object, ...]]],
    gold_manifest_sha256: str | None,
    issues: list[dict[str, str]],
    diagnostics: list[dict[str, str]],
    draft: bool,
) -> None:
    """Require complete, truthful assignments and a source-disjoint holdout."""

    if not _is_mapping(split):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_not_object",
                "split must be a JSON object",
                path="split",
                scope="split",
            ),
            draft=draft,
        )
        return
    status = split.get("status")
    if draft:
        if not _is_draft_status(status):
            diagnostics.append(
                _issue(
                    "split_status_not_draft",
                    "draft gold should use a draft split status while under review",
                    path="split.status",
                    scope="split",
                )
            )
    elif not _is_approved_status(status):
        issues.append(
            _issue(
                "split_not_approved",
                "approved gold needs a split that explicitly approves scoring",
                path="split.status",
                scope="split",
            )
        )
    expected_split_schema = DRAFT_SPLIT_SCHEMA if draft else APPROVED_SPLIT_SCHEMA
    if split.get("schema_version") != expected_split_schema:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_schema_version_invalid",
                "split schema_version is incompatible with its scoring status",
                path="split.schema_version",
                scope="split",
            ),
            draft=draft,
        )

    declared_gold_digest = _normalise_sha256(split.get("gold_manifest_sha256"))
    if declared_gold_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_gold_manifest_sha256_missing",
                "split needs a SHA-256 gold_manifest_sha256 binding",
                path="split.gold_manifest_sha256",
                scope="split",
            ),
            draft=draft,
        )
    elif gold_manifest_sha256 is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_manifest_digest_missing",
                "approved validation requires the supplied gold manifest raw-byte SHA-256",
                path="gold",
                scope="split",
            ),
            draft=draft,
        )
    elif declared_gold_digest != gold_manifest_sha256:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_gold_manifest_sha256_mismatch",
                "split gold_manifest_sha256 does not bind the supplied gold manifest",
                path="split.gold_manifest_sha256",
                scope="split",
            ),
            draft=draft,
        )

    assignments = split.get("assignments")
    if not isinstance(assignments, list) or not assignments:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_assignments_missing",
                "split must assign every reviewed gold family",
                path="split.assignments",
                scope="split",
            ),
            draft=draft,
        )
        return
    assignments_by_family: dict[str, Mapping[str, Any]] = {}
    for assignment_index, assignment in enumerate(assignments):
        path = f"split.assignments[{assignment_index}]"
        if not _is_mapping(assignment):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_assignment_not_object",
                    "each split assignment must be a JSON object",
                    path=path,
                    scope="split",
                ),
                draft=draft,
            )
            continue
        family_id = assignment.get("family_id")
        if not _is_nonempty_string(family_id):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_family_id_missing",
                    "each split assignment needs a non-empty family_id",
                    path=f"{path}.family_id",
                    scope="split",
                ),
                draft=draft,
            )
            continue
        family_id = str(family_id)
        if family_id in assignments_by_family:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_family_id_duplicate",
                    "a family may appear in exactly one split assignment",
                    path=f"{path}.family_id",
                    scope="split",
                ),
                draft=draft,
            )
            continue
        assignments_by_family[family_id] = assignment
        if not _is_nonempty_string(assignment.get("split")):
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_label_missing",
                    "each split assignment needs a non-empty split label",
                    path=f"{path}.split",
                    scope="split",
                ),
                draft=draft,
            )
        declared_source_keys = _safe_source_key_list(assignment.get("source_keys"))
        if declared_source_keys is None:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_source_keys_invalid",
                    "each assignment needs unique safe source_keys",
                    path=f"{path}.source_keys",
                    scope="split",
                ),
                draft=draft,
            )
        elif family_id in family_sources and declared_source_keys != family_sources[family_id]:
            _append_problem(
                issues,
                diagnostics,
                _issue(
                    "split_source_keys_mismatch",
                    "assignment source_keys must exactly match source keys derived from evidence atoms",
                    path=f"{path}.source_keys",
                    scope="split",
                ),
                draft=draft,
            )

    expected_families = set(family_sources)
    assigned_families = set(assignments_by_family)
    if missing := expected_families - assigned_families:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_missing_gold_family",
                "split does not assign every gold family",
                path="split.assignments",
                scope="split",
            ),
            draft=draft,
        )
    if unexpected := assigned_families - expected_families:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_unknown_gold_family",
                "split references a family absent from gold",
                path="split.assignments",
                scope="split",
            ),
            draft=draft,
        )

    all_source_keys = _safe_source_key_list(split.get("all_source_keys"))
    if all_source_keys is None or all_source_keys != set().union(*family_sources.values()):
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "split_all_source_keys_mismatch",
                "split all_source_keys must exactly match sources derived from gold atoms",
                path="split.all_source_keys",
                scope="split",
            ),
            draft=draft,
        )

    if draft:
        return
    holdout_families = {
        family_id
        for family_id, assignment in assignments_by_family.items()
        if assignment.get("split") == "holdout" and family_id in family_sources
    }
    if not holdout_families:
        issues.append(
            _issue(
                "split_missing_holdout",
                "approved Source gold needs at least one holdout family",
                path="split.assignments",
                scope="split",
            )
        )
        return
    for holdout_family in sorted(holdout_families):
        for other_family in sorted(expected_families - {holdout_family}):
            if family_source_identities[holdout_family].intersection(
                family_source_identities[other_family]
            ):
                issues.append(
                    _issue(
                        "holdout_source_overlap",
                        "a holdout family shares at least one source with another gold family",
                        path="split.assignments",
                        scope="split",
                    )
                )
                # One error proves the safety violation while keeping reports compact.
                return


def _validate_source_gold_with_verified_digests(
    gold: object,
    split: object,
    *,
    source_root: str | Path,
    frozen_source_manifest: object,
    frozen_source_manifest_sha256: str | None = None,
    gold_manifest_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate an evaluator-only gold/split pair after caller byte verification.

    This internal helper is deliberately reachable only from the bytes/file
    entry points below.  Those entry points derive its digest arguments from
    the exact bytes they strictly parsed.
    """

    issues: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    draft = _is_mapping(gold) and _is_draft_status(gold.get("status"))
    approved = _is_mapping(gold) and _is_approved_status(gold.get("status"))
    report: dict[str, Any] = {
        "schema": SOURCE_GOLD_VALIDATION_SCHEMA,
        "status": "invalid",
        "scoring_eligible": False,
        "counts": {
            "families": 0,
            "claim_groups": 0,
            "evidence_atoms": 0,
            "frozen_sources": 0,
        },
        "inputs": {
            "gold_status_category": _status_category(
                gold.get("status") if _is_mapping(gold) else None
            ),
            "split_status_category": _status_category(
                split.get("status") if _is_mapping(split) else None
            ),
            "gold_manifest_sha256": _normalise_sha256(gold_manifest_sha256),
            "split_manifest_sha256": _normalise_sha256(split_manifest_sha256),
            "frozen_source_manifest_sha256": _normalise_sha256(
                frozen_source_manifest_sha256
            ),
        },
        "issues": issues,
        "diagnostics": diagnostics,
    }
    if not _is_mapping(gold):
        issues.append(
            _issue(
                "gold_not_object",
                "gold must be a JSON object",
                path="gold",
                scope="gold",
            )
        )
        return report

    if draft:
        diagnostics.append(
            _issue(
                "gold_draft_not_scorable",
                "draft Source gold is diagnosable but cannot score an experiment",
                path="gold.status",
                scope="gold",
            )
        )
    elif not approved:
        issues.append(
            _issue(
                "gold_not_approved",
                "gold.status must explicitly approve scoring",
                path="gold.status",
                scope="gold",
            )
        )
    expected_gold_schema = DRAFT_GOLD_SCHEMA if draft else APPROVED_GOLD_SCHEMA
    if gold.get("schema_version") != expected_gold_schema:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_schema_version_invalid",
                "gold schema_version is incompatible with its scoring status",
                path="gold.schema_version",
                scope="gold",
            ),
            draft=draft,
        )
    if approved and not _is_approved_review_status(gold.get("review_status")):
        issues.append(
            _issue(
                "gold_review_status_not_approved",
                "approved gold needs an explicit approved review_status",
                path="gold.review_status",
                scope="gold",
            )
        )
    if gold.get("usable_for_scoring") is not True:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_not_usable_for_scoring",
                "approved gold must explicitly set usable_for_scoring to true",
                path="gold.usable_for_scoring",
                scope="gold",
            ),
            draft=draft,
        )

    frozen_manifest_digest = _normalise_sha256(frozen_source_manifest_sha256)
    declared_manifest_digest = _normalise_sha256(gold.get("source_manifest_sha256"))
    if declared_manifest_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_source_manifest_sha256_missing",
                "gold needs a SHA-256 source_manifest_sha256 binding",
                path="gold.source_manifest_sha256",
                scope="gold",
            ),
            draft=draft,
        )
    elif frozen_manifest_digest is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "frozen_source_manifest_digest_missing",
                "approved validation requires the frozen source manifest raw-byte SHA-256",
                path="source_manifest",
                scope="source_manifest",
            ),
            draft=draft,
        )
    elif declared_manifest_digest != frozen_manifest_digest:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "gold_source_manifest_sha256_mismatch",
                "gold source_manifest_sha256 does not bind the supplied frozen source manifest",
                path="gold.source_manifest_sha256",
                scope="gold",
            ),
            draft=draft,
        )

    frozen_sources = _parse_frozen_sources(
        frozen_source_manifest,
        issues=issues,
        diagnostics=diagnostics,
        draft=draft,
    )
    report["counts"]["frozen_sources"] = len(frozen_sources)
    requested_source_root, source_root_rejection = _direct_local_path_preflight(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if requested_source_root is None:
        # This fallback is never used for source I/O: a rejection below keeps
        # ``root_snapshot_candidate`` empty.  It merely preserves a safe path
        # value for diagnostic-only control flow.
        requested_source_root = Path(".")
    root_snapshot_candidate = (
        None
        if source_root_rejection is not None
        else _path_snapshot(requested_source_root)
    )
    source_root_path = (
        root_snapshot_candidate[0]
        if root_snapshot_candidate is not None
        else requested_source_root
    )
    source_root_snapshot = (
        root_snapshot_candidate
        if _snapshot_is_directory(root_snapshot_candidate)
        else None
    )
    if source_root_rejection is not None:
        message_by_code = {
            "source_root_unc_not_allowed": "source_root must not use a UNC network path",
            "source_root_remote_drive_not_allowed": "source_root must not use a remote mapped drive",
            "source_root_locality_unverifiable": "source_root localness could not be verified",
            "source_root_link_or_junction_not_allowed": "source_root must be a direct local directory, not a link or junction",
            "source_root_unavailable": "source_root must be a local path",
        }
        _append_problem(
            issues,
            diagnostics,
            _issue(
                source_root_rejection,
                message_by_code.get(source_root_rejection, "source_root is not a permitted local directory"),
                path="source_root",
                scope="source_manifest",
            ),
            draft=draft,
        )
    elif source_root_snapshot is None:
        _append_problem(
            issues,
            diagnostics,
            _issue(
                "source_root_unavailable",
                "source_root must be an existing directory",
                path="source_root",
                scope="source_manifest",
            ),
            draft=draft,
        )

    source_identities = (
        _validate_frozen_source_identities(
            frozen_sources,
            source_root=source_root_path,
            source_root_snapshot=source_root_snapshot,
            issues=issues,
            diagnostics=diagnostics,
            draft=draft,
        )
        if source_root_snapshot is not None
        else {}
    )
    source_cache: dict[str, tuple[str | None, object | None, _PathSnapshot | None]] = {}
    _validate_frozen_source_contents(
        frozen_sources,
        source_root=source_root_path,
        source_root_snapshot=source_root_snapshot,
        source_identities=source_identities,
        source_cache=source_cache,
        issues=issues,
        diagnostics=diagnostics,
        draft=draft,
    )
    (
        family_sources,
        family_source_identities,
        family_ids,
        group_count,
        atom_count,
    ) = _validate_gold_structure(
        gold,
        source_root=source_root_path,
        source_root_snapshot=source_root_snapshot,
        frozen_sources=frozen_sources,
        source_identities=source_identities,
        source_cache=source_cache,
        issues=issues,
        diagnostics=diagnostics,
        draft=draft,
    )
    report["counts"].update(
        {
            "families": len(family_ids),
            "claim_groups": group_count,
            "evidence_atoms": atom_count,
        }
    )
    _validate_split(
        split,
        family_sources=family_sources,
        family_source_identities=family_source_identities,
        gold_manifest_sha256=_normalise_sha256(gold_manifest_sha256),
        issues=issues,
        diagnostics=diagnostics,
        draft=draft,
    )
    _validate_source_cache_stability(
        source_cache,
        source_root=source_root_path,
        source_root_snapshot=source_root_snapshot,
        issues=issues,
        diagnostics=diagnostics,
        draft=draft,
    )

    if draft:
        report["status"] = "draft"
    elif approved and not issues:
        report["status"] = "ready"
        report["scoring_eligible"] = True
    return report


def validate_source_gold_bytes(
    gold_bytes: bytes,
    split_bytes: bytes,
    *,
    source_root: str | Path,
    frozen_source_manifest_bytes: bytes,
) -> dict[str, Any]:
    """Validate exact artifact bytes plus frozen local Source files.

    This is the scoring-capable API.  It parses every artifact strictly and
    derives every manifest digest from the exact supplied bytes, so a caller
    cannot substitute an unrelated object/digest pair after loading.  It does
    not write the source tree, knowledge base, or any provider-facing state.
    """

    if not all(
        isinstance(value, (bytes, bytearray, memoryview))
        for value in (gold_bytes, split_bytes, frozen_source_manifest_bytes)
    ):
        raise SourceGoldValidationInputError("artifact inputs must be byte sequences")
    # ``bytes`` copies mutable bytearrays/memoryviews before parsing or hashing.
    gold_bytes = bytes(gold_bytes)
    split_bytes = bytes(split_bytes)
    frozen_source_manifest_bytes = bytes(frozen_source_manifest_bytes)
    gold = _parse_artifact_json(gold_bytes)
    split = _parse_artifact_json(split_bytes)
    frozen_source_manifest = _parse_artifact_json(frozen_source_manifest_bytes)
    return _validate_source_gold_with_verified_digests(
        gold,
        split,
        source_root=source_root,
        frozen_source_manifest=frozen_source_manifest,
        frozen_source_manifest_sha256=sha256(frozen_source_manifest_bytes).hexdigest(),
        gold_manifest_sha256=sha256(gold_bytes).hexdigest(),
        split_manifest_sha256=sha256(split_bytes).hexdigest(),
    )


def validate_source_gold(
    gold: object,
    split: object,
    *,
    source_root: str | Path,
    frozen_source_manifest: object,
    frozen_source_manifest_sha256: str | None = None,
    gold_manifest_sha256: str | None = None,
    split_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Diagnose in-memory objects without ever authorizing scoring.

    Object values plus caller-supplied digest strings cannot prove the digests
    came from an immutable artifact.  Use :func:`validate_source_gold_bytes`
    or :func:`validate_source_gold_files` for the formal gate.  This entry
    remains useful to editors because it performs the same local diagnostics,
    but it deliberately fails closed before returning ``ready``.
    """

    report = _validate_source_gold_with_verified_digests(
        gold,
        split,
        source_root=source_root,
        frozen_source_manifest=frozen_source_manifest,
        frozen_source_manifest_sha256=frozen_source_manifest_sha256,
        gold_manifest_sha256=gold_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
    )
    if report["scoring_eligible"]:
        report["issues"].append(
            _issue(
                "raw_artifact_binding_unavailable",
                "in-memory objects cannot authorize scoring without exact artifact bytes",
                path="gold",
                scope="setup",
            )
        )
        report["status"] = "invalid"
        report["scoring_eligible"] = False
    return report


def validate_source_gold_files(
    gold_path: str | Path,
    split_path: str | Path,
    *,
    source_root: str | Path,
    frozen_source_manifest_path: str | Path,
) -> dict[str, Any]:
    """Load only local JSON files, validate them, and return a text-safe report."""

    (
        gold_direct_path,
        split_direct_path,
        frozen_source_manifest_direct_path,
    ) = _require_direct_local_artifact_inputs(
        gold_path=gold_path,
        split_path=split_path,
        frozen_source_manifest_path=frozen_source_manifest_path,
    )
    direct_source_root, source_root_rejection = _direct_local_path_preflight(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if source_root_rejection is not None or direct_source_root is None:
        raise SourceGoldValidationInputError(
            "source_root is not a permitted direct local directory"
        )
    return validate_source_gold_bytes(
        _read_json_bytes(gold_direct_path, label="gold"),
        _read_json_bytes(split_direct_path, label="split"),
        source_root=direct_source_root,
        frozen_source_manifest_bytes=_read_json_bytes(
            frozen_source_manifest_direct_path,
            label="frozen_source_manifest",
        ),
    )


def _report_output_preflight(path: str | Path) -> tuple[Path | None, str | None]:
    """Return a fixed direct-local report target without resolving it.

    A missing *leaf* is fine only when its already-existing parent is a direct
    local directory.  Creating an arbitrary missing parent would turn a later
    ``mkdir`` into an implicit traversal of a network share, symlink, or
    junction, so formal CLI output deliberately does not support that form.
    """

    direct_path, rejection = _direct_local_path_preflight(
        path,
        prefix="output",
        expect_directory=False,
        allow_missing_leaf=True,
    )
    if rejection is not None or direct_path is None or not direct_path.name:
        return None, rejection or "output_unavailable"
    try:
        metadata = direct_path.lstat()
    except FileNotFoundError:
        # The leaf does not exist yet.  Its parent must already be a direct
        # local directory; do not call mkdir to manufacture a path under an
        # unverified parent.
        parent_path, parent_rejection = _direct_local_path_preflight(
            direct_path.parent,
            prefix="output_parent",
            expect_directory=True,
        )
        if parent_rejection is not None or parent_path is None:
            return None, parent_rejection or "output_parent_unavailable"
        return direct_path, None
    except (OSError, RuntimeError):
        return None, "output_unavailable"
    if not S_ISREG(metadata.st_mode):
        return None, "output_unavailable"
    return direct_path, None


def _report_output_rejection_code(path: str | Path) -> str | None:
    """Return only the rejection part of :func:`_report_output_preflight`."""

    _direct_path, rejection = _report_output_preflight(path)
    return rejection


def _resolve_output_guard_path(path: Path) -> Path | None:
    """Resolve a potential report target without requiring it to exist."""

    try:
        return path.resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _normalised_path_key(path: Path) -> str:
    """Create a platform-aware comparison key for already-resolved paths."""

    return os.path.normcase(os.path.normpath(str(path)))


def _path_is_within(candidate: Path, root: Path) -> bool:
    """Check containment after resolving links/junctions and normalizing case."""

    candidate_key = _normalised_path_key(candidate)
    root_key = _normalised_path_key(root)
    try:
        return os.path.commonpath((candidate_key, root_key)) == root_key
    except ValueError:
        # Different drives (on Windows) cannot be nested.
        return False


def _safe_report_output_path(
    path: Path,
    *,
    artifact_paths: Sequence[Path],
    source_root: Path,
) -> bool:
    """Refuse report targets that could overwrite an input or source snapshot.

    The direct resolved-path comparison protects normal aliases and links;
    stable physical identities additionally cover hard links.  If an existing
    candidate or input has no stable identity, the CLI cannot prove it is not
    an alias and rejects the file output rather than risking an overwrite.
    """

    # Every path below must pass a non-resolving direct-local check before this
    # helper calls ``resolve`` for alias/overwrite comparisons.  Subsequent
    # work uses the fixed absolute forms, not caller-relative spellings.
    direct_output_path, output_rejection = _report_output_preflight(path)
    if output_rejection is not None or direct_output_path is None:
        return False
    direct_source_root, source_root_rejection = _direct_local_path_preflight(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if source_root_rejection is not None or direct_source_root is None:
        return False
    direct_artifact_paths: list[Path] = []
    for artifact_path in artifact_paths:
        direct_artifact_path, artifact_rejection = _direct_local_path_preflight(
            artifact_path,
            prefix="artifact_input",
            expect_directory=False,
        )
        if artifact_rejection is not None or direct_artifact_path is None:
            return False
        direct_artifact_paths.append(direct_artifact_path)
    candidate = _resolve_output_guard_path(direct_output_path)
    resolved_source_root = _resolve_output_guard_path(direct_source_root)
    if candidate is None or resolved_source_root is None:
        return False
    if _path_is_within(candidate, resolved_source_root):
        return False

    try:
        candidate_exists = candidate.exists()
    except OSError:
        return False
    if candidate_exists and not candidate.is_file():
        return False
    candidate_identity = _source_file_identity(candidate) if candidate_exists else None
    if candidate_exists and candidate_identity is None:
        return False

    for artifact_path in direct_artifact_paths:
        artifact = _resolve_output_guard_path(artifact_path)
        if artifact is None:
            return False
        if _normalised_path_key(candidate) == _normalised_path_key(artifact):
            return False
        try:
            if not artifact.exists():
                return False
        except OSError:
            return False
        artifact_identity = _source_file_identity(artifact)
        if artifact_identity is None:
            return False
        if candidate_identity is not None and candidate_identity == artifact_identity:
            return False
    return True


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    artifact_paths: Sequence[Path],
    source_root: Path,
) -> bool:
    """Safely write a text-safe report, returning false on an unsafe target."""

    direct_output_path, output_rejection = _report_output_preflight(path)
    direct_source_root, source_root_rejection = _direct_local_path_preflight(
        source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if (
        output_rejection is not None
        or direct_output_path is None
        or source_root_rejection is not None
        or direct_source_root is None
    ):
        return False
    direct_artifact_paths: list[Path] = []
    for artifact_path in artifact_paths:
        direct_artifact_path, artifact_rejection = _direct_local_path_preflight(
            artifact_path,
            prefix="artifact_input",
            expect_directory=False,
        )
        if artifact_rejection is not None or direct_artifact_path is None:
            return False
        direct_artifact_paths.append(direct_artifact_path)
    # The output preflight requires an existing verified direct-local parent.
    # Do not manufacture missing parents here: doing so could create a report
    # path underneath a substituted network/link target.
    if not _safe_report_output_path(
        direct_output_path,
        artifact_paths=direct_artifact_paths,
        source_root=direct_source_root,
    ):
        return False
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            "wb",
            dir=direct_output_path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if not _safe_report_output_path(
            direct_output_path,
            artifact_paths=direct_artifact_paths,
            source_root=direct_source_root,
        ):
            return False
        os.replace(temporary, direct_output_path)
        return True
    except OSError:
        return False
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True, help="Source-gold JSON")
    parser.add_argument("--split", type=Path, required=True, help="gold split JSON")
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="root directory containing frozen corpus-relative source files",
    )
    parser.add_argument(
        "--frozen-source-manifest",
        type=Path,
        required=True,
        help="frozen source-key / source-file-hash JSON manifest",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="validation report JSON path, or '-' for stdout",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    # Artifact paths must be classified before any output guard resolves one of
    # them.  In particular, a UNC or mapped-drive spelling must never be
    # resolved merely because the caller requested ``--output``.  Retain and
    # pass only the resulting fixed absolute paths from this point onward.
    try:
        (
            gold_path,
            split_path,
            frozen_source_manifest_path,
        ) = _require_direct_local_artifact_inputs(
            gold_path=args.gold,
            split_path=args.split,
            frozen_source_manifest_path=args.frozen_source_manifest,
        )
    except SourceGoldValidationInputError:
        return 2
    source_root, source_root_rejection = _direct_local_path_preflight(
        args.source_root,
        prefix="source_root",
        expect_directory=True,
    )
    if source_root_rejection is not None or source_root is None:
        return 2
    output_path: Path | None = None
    if args.output != "-":
        output_path, output_rejection = _report_output_preflight(args.output)
        if output_rejection is not None or output_path is None:
            return 2
    artifact_paths = (gold_path, split_path, frozen_source_manifest_path)
    # Do this before loading inputs: a report must never overwrite the gold,
    # split, frozen manifest, or a file inside the frozen source snapshot.
    if output_path is not None and not _safe_report_output_path(
        output_path,
        artifact_paths=artifact_paths,
        source_root=source_root,
    ):
        return 2
    try:
        report = validate_source_gold_files(
            gold_path,
            split_path,
            source_root=source_root,
            frozen_source_manifest_path=frozen_source_manifest_path,
        )
    except SourceGoldValidationInputError as exc:
        report = {
            "schema": SOURCE_GOLD_VALIDATION_SCHEMA,
            "status": "invalid",
            "scoring_eligible": False,
            "counts": {
                "families": 0,
                "claim_groups": 0,
                "evidence_atoms": 0,
                "frozen_sources": 0,
            },
            "inputs": {},
            "issues": [
                _issue(
                    "input_load_failed",
                    str(exc),
                    path="cli",
                    scope="setup",
                )
            ],
            "diagnostics": [],
        }
    if args.output == "-":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if not _atomic_write_json(
            output_path,
            report,
            artifact_paths=artifact_paths,
            source_root=source_root,
        ):
            return 2
    return 0 if report["scoring_eligible"] else 2


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
