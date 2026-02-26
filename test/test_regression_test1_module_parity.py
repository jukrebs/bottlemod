"""Regression Tets1 module parity test."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
VANILLA_DIR = ROOT / "bottlemod_vanilla"
VANILLA_PAPER_FIGURES = VANILLA_DIR / "paper_figures_general.py"


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _temporary_sys_modules(overrides: dict[str, ModuleType]):
    previous = {key: sys.modules.get(key) for key in overrides}
    sys.modules.update(overrides)
    try:
        yield
    finally:
        for key, old in previous.items():
            if old is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = old


def _build_vanilla_module_aliases() -> dict[str, ModuleType]:
    vanilla_ppoly = _load_module_from_path("_vanilla_ppoly", VANILLA_DIR / "ppoly.py")
    with _temporary_sys_modules({"ppoly": vanilla_ppoly}):
        vanilla_func = _load_module_from_path("_vanilla_func", VANILLA_DIR / "func.py")
    with _temporary_sys_modules({"ppoly": vanilla_ppoly, "func": vanilla_func}):
        vanilla_task = _load_module_from_path("_vanilla_task", VANILLA_DIR / "task.py")

    return {
        "ppoly": vanilla_ppoly,
        "func": vanilla_func,
        "task": vanilla_task,
    }


def _build_bottlemod_module_aliases() -> dict[str, ModuleType]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    return {
        "ppoly": importlib.import_module("bottlemod.ppoly"),
        "func": importlib.import_module("bottlemod.func"),
        "task": importlib.import_module("bottlemod.task"),
    }


def _run_vanilla_test1_with_aliases(
    alias_modules: dict[str, ModuleType],
) -> tuple[str, Exception | None]:
    # paper_figures_general.py does: from task import *, from func import *, from ppoly import *
    with _temporary_sys_modules(alias_modules):
        paper_figures_module = _load_module_from_path(
            f"_paper_figures_general_{id(alias_modules)}", VANILLA_PAPER_FIGURES
        )
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            try:
                paper_figures_module.Test1()
                return buffer.getvalue(), None
            except Exception as exc:  # pylint: disable=broad-except
                return buffer.getvalue(), exc


def _normalize_output(output: str) -> str:
    lines = [line.rstrip() for line in output.strip().splitlines()]
    return "\n".join(lines)


class TestRegressionTest1ModuleParity(unittest.TestCase):
    def test_test1_output_matches_bottlemod_and_vanilla(self):
        vanilla_output, vanilla_error = _run_vanilla_test1_with_aliases(
            _build_vanilla_module_aliases()
        )
        bottlemod_output, bottlemod_error = _run_vanilla_test1_with_aliases(
            _build_bottlemod_module_aliases()
        )

        self.assertIsNone(vanilla_error, f"vanilla `Test1()` raised: {vanilla_error}")
        self.assertIsNone(
            bottlemod_error,
            f"bottlemod `Test1()` raised while vanilla did not: {bottlemod_error}",
        )

        self.assertEqual(
            _normalize_output(bottlemod_output),
            _normalize_output(vanilla_output),
            "`Test1()` output differs between `bottlemod` and `bottlemod_vanilla`.",
        )


if __name__ == "__main__":
    unittest.main()
