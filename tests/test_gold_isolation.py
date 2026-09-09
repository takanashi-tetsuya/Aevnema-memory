"""Keep production code independent from benchmark and gold-data artifacts.

This is deliberately an AST-based guard instead of a text search: comments and
trace-contract metadata may name a gold artifact without making production code
depend on it.  Only imports and constant filesystem paths used by I/O/path
construction are considered dependencies.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path
import unittest


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PRODUCTION_ROOT = _PROJECT_ROOT / "src" / "memory_demo"
_FORBIDDEN_MODULE_ROOTS = frozenset({"benchmarks", "validation"})
_PATH_CONSTRUCTOR_NAMES = frozenset(
    {"Path", "PurePath", "PosixPath", "WindowsPath"}
)
_MAX_STATIC_PATH_VARIANTS = 32


def _normalise_path(value: str) -> str:
    return value.replace("\\", "/").lower()


def _is_forbidden_path(value: str) -> bool:
    """Return whether a constant path identifies a gold/benchmark artifact."""

    normalised = _normalise_path(value)
    if "gold_manifest" in normalised or "split_manifest" in normalised:
        return True

    # Treat directory names as path components.  This covers both
    # ``validation/file.json`` and ``Path("validation") / "file.json"`` while
    # not treating ordinary prose such as ``"validation/"`` as a dependency
    # unless it is actually handed to a filesystem operation below.
    components = [component for component in normalised.split("/") if component]
    return any(component in _FORBIDDEN_MODULE_ROOTS for component in components)


def _join_path_values(parts: Iterable[set[str]]) -> set[str]:
    """Join finite sets of constant path parts without unbounded expansion."""

    values = {""}
    for part_values in parts:
        if not part_values:
            return set()
        next_values: set[str] = set()
        for prefix in values:
            for part in part_values:
                if prefix:
                    combined = f"{prefix.rstrip('/\\')}/{part.lstrip('/\\')}"
                else:
                    combined = part
                next_values.add(combined)
                if len(next_values) > _MAX_STATIC_PATH_VARIANTS:
                    return set()
        values = next_values
    return values


class _GoldIsolationVisitor(ast.NodeVisitor):
    """Find static production dependencies on benchmark/gold-only inputs."""

    def __init__(self, filename: Path) -> None:
        self.filename = filename
        self.violations: list[str] = []
        self._value_scopes: list[dict[str, set[str]]] = [{}]
        self._pathlib_scopes: list[set[str]] = [{"pathlib"}]
        self._json_scopes: list[set[str]] = [{"json"}]
        self._io_scopes: list[set[str]] = [{"io"}]
        self._path_constructor_scopes: list[set[str]] = [
            set(_PATH_CONSTRUCTOR_NAMES)
        ]
        self._path_join_scopes: list[set[str]] = [set()]

    @property
    def _values(self) -> dict[str, set[str]]:
        return self._value_scopes[-1]

    @property
    def _pathlib_names(self) -> set[str]:
        return self._pathlib_scopes[-1]

    @property
    def _json_names(self) -> set[str]:
        return self._json_scopes[-1]

    @property
    def _io_names(self) -> set[str]:
        return self._io_scopes[-1]

    @property
    def _path_constructor_names(self) -> set[str]:
        return self._path_constructor_scopes[-1]

    @property
    def _path_join_names(self) -> set[str]:
        return self._path_join_scopes[-1]

    def _push_scope(self) -> None:
        self._value_scopes.append(dict(self._values))
        self._pathlib_scopes.append(set(self._pathlib_names))
        self._json_scopes.append(set(self._json_names))
        self._io_scopes.append(set(self._io_names))
        self._path_constructor_scopes.append(set(self._path_constructor_names))
        self._path_join_scopes.append(set(self._path_join_names))

    def _pop_scope(self) -> None:
        self._value_scopes.pop()
        self._pathlib_scopes.pop()
        self._json_scopes.pop()
        self._io_scopes.pop()
        self._path_constructor_scopes.pop()
        self._path_join_scopes.pop()

    def _record(self, node: ast.AST, detail: str) -> None:
        self.violations.append(
            f"{self.filename}:{getattr(node, 'lineno', '?')}: {detail}"
        )

    @staticmethod
    def _module_is_forbidden(module: str | None) -> bool:
        if not module:
            return False
        return module.split(".", 1)[0] in _FORBIDDEN_MODULE_ROOTS

    @staticmethod
    def _attribute_base_name(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _is_path_constructor(self, function: ast.AST) -> bool:
        if isinstance(function, ast.Name):
            return function.id in self._path_constructor_names
        if isinstance(function, ast.Attribute):
            return (
                function.attr in _PATH_CONSTRUCTOR_NAMES
                and self._attribute_base_name(function.value) in self._pathlib_names
            )
        return False

    def _is_path_join(self, function: ast.AST) -> bool:
        if isinstance(function, ast.Name):
            return function.id in self._path_join_names
        if not isinstance(function, ast.Attribute):
            return False
        if function.attr == "joinpath":
            return True
        if function.attr != "join" or not isinstance(function.value, ast.Attribute):
            return False
        return (
            function.value.attr == "path"
            and self._attribute_base_name(function.value.value) == "os"
        )

    def _is_open_call(self, function: ast.AST) -> bool:
        if isinstance(function, ast.Name):
            return function.id == "open"
        if not isinstance(function, ast.Attribute) or function.attr != "open":
            return False
        base = self._attribute_base_name(function.value)
        return base in self._io_names or bool(self._constant_path_values(function.value))

    def _is_json_load(self, function: ast.AST) -> bool:
        return (
            isinstance(function, ast.Attribute)
            and function.attr == "load"
            and self._attribute_base_name(function.value) in self._json_names
        )

    @staticmethod
    def _first_path_argument(
        node: ast.Call, *, keyword_name: str
    ) -> ast.AST | None:
        if node.args:
            return node.args[0]
        for keyword in node.keywords:
            if keyword.arg == keyword_name:
                return keyword.value
        return None

    def _constant_path_values(self, node: ast.AST) -> set[str]:
        """Evaluate a small, safe subset of compile-time path expressions."""

        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return {node.value}
        if isinstance(node, ast.Name):
            return set(self._values.get(node.id, set()))
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for part in node.values:
                if not isinstance(part, ast.Constant) or not isinstance(part.value, str):
                    return set()
                parts.append(part.value)
            return {"".join(parts)}
        if isinstance(node, ast.IfExp):
            return self._constant_path_values(node.body) | self._constant_path_values(
                node.orelse
            )
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = self._constant_path_values(node.left)
            right = self._constant_path_values(node.right)
            return {
                f"{left_value}{right_value}"
                for left_value in left
                for right_value in right
                if len(left) * len(right) <= _MAX_STATIC_PATH_VARIANTS
            }
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            return _join_path_values(
                (
                    self._constant_path_values(node.left),
                    self._constant_path_values(node.right),
                )
            )
        if isinstance(node, ast.Call):
            if self._is_path_constructor(node.func) or self._is_path_join(node.func):
                return _join_path_values(
                    self._constant_path_values(argument) for argument in node.args
                )
            if self._is_open_call(node.func):
                argument = self._first_path_argument(node, keyword_name="file")
                if argument is not None:
                    return self._constant_path_values(argument)
        return set()

    def _record_forbidden_values(
        self, node: ast.AST, *, context: str, values: set[str]
    ) -> None:
        for value in sorted(values):
            if _is_forbidden_path(value):
                self._record(node, f"{context} references forbidden path {value!r}")

    def _bind(self, target: ast.AST, values: set[str]) -> None:
        if isinstance(target, ast.Name):
            self._values[target.id] = set(values)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._bind(item, set())

    def _merge_branch_values(
        self, before: dict[str, set[str]], after: dict[str, set[str]]
    ) -> dict[str, set[str]]:
        keys = set(before) | set(after)
        return {
            key: set(before.get(key, set())) | set(after.get(key, set()))
            for key in keys
        }

    def _visit_branch(self, statements: list[ast.stmt]) -> tuple[
        dict[str, set[str]], set[str], set[str], set[str], set[str], set[str]
    ]:
        self._push_scope()
        try:
            for statement in statements:
                self.visit(statement)
            return (
                dict(self._values),
                set(self._pathlib_names),
                set(self._json_names),
                set(self._io_names),
                set(self._path_constructor_names),
                set(self._path_join_names),
            )
        finally:
            self._pop_scope()

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if self._module_is_forbidden(alias.name):
                self._record(node, f"imports forbidden module {alias.name!r}")
            local_name = alias.asname or alias.name.split(".", 1)[0]
            if alias.name == "pathlib":
                self._pathlib_names.add(local_name)
            elif alias.name == "json":
                self._json_names.add(local_name)
            elif alias.name == "io":
                self._io_names.add(local_name)
            elif alias.name == "os":
                # The canonical ``os.path.join`` spelling is handled directly.
                pass

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._module_is_forbidden(node.module):
            self._record(node, f"imports forbidden module {node.module!r}")
        if node.module is None:
            for alias in node.names:
                if alias.name in _FORBIDDEN_MODULE_ROOTS:
                    self._record(node, f"imports forbidden module {alias.name!r}")
        for alias in node.names:
            local_name = alias.asname or alias.name
            if node.module == "pathlib" and alias.name in _PATH_CONSTRUCTOR_NAMES:
                self._path_constructor_names.add(local_name)
            elif node.module == "os.path" and alias.name == "join":
                self._path_join_names.add(local_name)

    def visit_Assign(self, node: ast.Assign) -> None:
        self.visit(node.value)
        values = self._constant_path_values(node.value)
        for target in node.targets:
            self._bind(target, values)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self.visit(node.value)
            self._bind(node.target, self._constant_path_values(node.value))

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self.visit(node.value)
        self._bind(node.target, set())

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.visit(node.value)
        self._bind(node.target, self._constant_path_values(node.value))

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        before_values = dict(self._values)
        before_pathlib = set(self._pathlib_names)
        before_json = set(self._json_names)
        before_io = set(self._io_names)
        before_constructors = set(self._path_constructor_names)
        before_joins = set(self._path_join_names)
        body = self._visit_branch(node.body)
        otherwise = self._visit_branch(node.orelse)

        self._values.clear()
        self._values.update(
            self._merge_branch_values(
                self._merge_branch_values(before_values, body[0]), otherwise[0]
            )
        )
        self._pathlib_names.clear()
        self._pathlib_names.update(before_pathlib | body[1] | otherwise[1])
        self._json_names.clear()
        self._json_names.update(before_json | body[2] | otherwise[2])
        self._io_names.clear()
        self._io_names.update(before_io | body[3] | otherwise[3])
        self._path_constructor_names.clear()
        self._path_constructor_names.update(before_constructors | body[4] | otherwise[4])
        self._path_join_names.clear()
        self._path_join_names.update(before_joins | body[5] | otherwise[5])

    def _visit_scoped_body(self, node: ast.AST, body: list[ast.stmt]) -> None:
        for decorator in getattr(node, "decorator_list", []):
            self.visit(decorator)
        self._push_scope()
        try:
            for statement in body:
                self.visit(statement)
        finally:
            self._pop_scope()

    def _visit_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        arguments = node.args
        for argument in (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        ):
            if argument.annotation is not None:
                self.visit(argument.annotation)
        if arguments.vararg is not None and arguments.vararg.annotation is not None:
            self.visit(arguments.vararg.annotation)
        if arguments.kwarg is not None and arguments.kwarg.annotation is not None:
            self.visit(arguments.kwarg.annotation)
        for default in (*arguments.defaults, *arguments.kw_defaults):
            if default is not None:
                self.visit(default)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_signature(node)
        self._visit_scoped_body(node, node.body)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function_signature(node)
        self._visit_scoped_body(node, node.body)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword.value)
        self._visit_scoped_body(node, node.body)

    def visit_Call(self, node: ast.Call) -> None:
        if self._is_path_constructor(node.func) or self._is_path_join(node.func):
            self._record_forbidden_values(
                node,
                context="path construction",
                values=self._constant_path_values(node),
            )
        elif self._is_open_call(node.func):
            argument = self._first_path_argument(node, keyword_name="file")
            if argument is not None:
                self._record_forbidden_values(
                    node,
                    context="file open",
                    values=self._constant_path_values(argument),
                )
        elif (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"read_text", "read_bytes"}
        ):
            self._record_forbidden_values(
                node,
                context=f"{node.func.attr} call",
                values=self._constant_path_values(node.func.value),
            )
        elif self._is_json_load(node.func):
            argument = self._first_path_argument(node, keyword_name="fp")
            if argument is not None:
                self._record_forbidden_values(
                    node,
                    context="json.load call",
                    values=self._constant_path_values(argument),
                )
        self.generic_visit(node)


def _scan_source(source: str, filename: Path) -> list[str]:
    tree = ast.parse(source, filename=str(filename))
    visitor = _GoldIsolationVisitor(filename)
    visitor.visit(tree)
    return visitor.violations


def _scan_production_modules() -> list[str]:
    violations: list[str] = []
    for path in sorted(_PRODUCTION_ROOT.rglob("*.py")):
        violations.extend(_scan_source(path.read_text(encoding="utf-8"), path))
    return violations


class GoldIsolationTests(unittest.TestCase):
    def test_production_source_has_no_gold_or_benchmark_dependencies(self):
        violations = _scan_production_modules()

        self.assertEqual([], violations, "\n".join(violations))

    def test_ast_guard_rejects_imports_and_constant_io_paths(self):
        source = '''
import benchmarks.import_corpus
from validation import split_manifest
from pathlib import Path
import json

base = Path("validation")
payload = base / "gold_manifest.json"
open("benchmarks/cases.json", encoding="utf-8")
Path("split_manifest.json").read_text(encoding="utf-8")
json.load(open(payload, encoding="utf-8"))
open(file="validation/keyword.json", encoding="utf-8")

def delayed_default(path=Path("benchmarks/default.json")):
    return path
'''

        violations = _scan_source(source, Path("fixture.py"))

        self.assertGreaterEqual(len(violations), 8)
        rendered = "\n".join(violations)
        self.assertIn("imports forbidden module 'benchmarks.import_corpus'", rendered)
        self.assertIn("imports forbidden module 'validation'", rendered)
        self.assertIn("path construction", rendered)
        self.assertIn("file open", rendered)
        self.assertIn("read_text call", rendered)

    def test_comments_and_trace_schema_metadata_are_not_dependencies(self):
        source = '''
# open("benchmarks/comment-only.json")
TRACE_SCHEMA_METADATA = {
    "gold_manifest": "gold_manifest is an optional provenance field",
    "description": "validation/ is a documentation example only",
}
'''

        self.assertEqual([], _scan_source(source, Path("trace_metadata.py")))


if __name__ == "__main__":
    unittest.main()
