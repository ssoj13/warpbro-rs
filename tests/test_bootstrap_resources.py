"""CPU resource-receipt regression tests; no compiler or GPU is launched.

Run: python -m unittest discover -s tests -p test_bootstrap_resources.py -v
Optionally set WARPBRO_TEST_PTXAS_RECEIPT to a real receipt JSON and
WARPBRO_TEST_PTX to its exact input PTX (otherwise nearby PTX is located).
"""

import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import bootstrap


PTX = (
    b".version 8.0\n.target sm_86\n.address_size 64\n"
    b".visible .func helper() {ret;}\n"
    b".visible .entry kernel() {call.uni helper, (); ret;}\n"
)
LOG = """ptxas info : Compiling entry function 'kernel' for 'sm_86'
ptxas info : Function properties for kernel
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info : Used 8 registers
ptxas info : Function properties for helper
  0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
"""


class ResourceReceiptTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.ptx = self.root / "fixture.ptx"
        self.ptx.write_bytes(PTX)
        self.tool = self.root / "ptxas.exe"
        self.tool.write_bytes(b"actual fixture tool bytes")
        self.sha = hashlib.sha256(PTX).hexdigest()
        self.entries, self.calls = bootstrap.ptx_call_graph(PTX.decode())
        self.metadata = {
            "schema": 1,
            "kind": "ptxas-assembly-receipt",
            "module": "fixture",
            "host_target": "fixture-host",
            "arch": "sm_86",
            "ptx_name": "fixture.ptx",
            "ptx_sha256": self.sha,
            "ptxas_path": str(self.tool),
            "ptxas_sha256": hashlib.sha256(self.tool.read_bytes()).hexdigest(),
            "options": [
                "--gpu-name=sm_86", "--fmad=true", "--split-compile=4", "--verbose",
            ],
            "entries": ["kernel"],
            "raw_log": "sm_86.ptxas.log",
            "assembly_succeeded": True,
            "resource_coverage_complete": None,
        }
        self.receipt = self.write_receipt(self.metadata, LOG)

    def write_receipt(self, metadata, log, index=0):
        folder = self.root / f"fixture.host.10.{index}.ptxas-report"
        folder.mkdir(exist_ok=True)
        path = folder / "sm_86.ptxas.json"
        path.write_text(json.dumps(metadata), encoding="utf-8")
        (folder / "sm_86.ptxas.log").write_text(log, encoding="utf-8")
        return path

    def retained(self):
        return bootstrap.retained_kernel_stats(
            self.ptx, "sm_86", self.sha, self.entries, self.calls,
        )

    def assert_rejected(self, metadata=None, log=LOG):
        self.write_receipt(metadata if metadata is not None else self.metadata, log)
        with self.assertRaises((RuntimeError, ValueError, OSError)):
            self.retained()

    def invoke(self, **kwargs):
        arguments = argparse.Namespace(
            rest=["sm_86"], build=False, lines=False, assemble=False,
            no_spills=True, strict=False,
        )
        for key, value in kwargs.items():
            setattr(arguments, key, value)
        with (
            patch.object(bootstrap, "ROOT_DIR", self.root),
            patch.object(bootstrap, "KERNEL_PTX", self.ptx),
            patch.object(bootstrap, "KERNEL_REPORT_DIR", self.root / "kernels"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return bootstrap.run_kernels(arguments)

    def test_valid_complete_helper_receipt(self):
        statistics, provenance = self.retained()
        self.assertEqual(statistics["helper"]["st"], 0)
        self.assertEqual(provenance["resource_source"], "production-aot-receipt")

    def test_equivalent_repeated_builds(self):
        self.write_receipt(self.metadata, LOG + "ptxas info : Compile time = 1 ms\n", 1)
        self.assertEqual(len(self.retained()[1]["receipts"]), 2)

    def test_conflicting_policy_rejected(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["options"][1] = "--fmad=false"
        self.write_receipt(metadata, LOG, 1)
        with self.assertRaisesRegex(RuntimeError, "Conflicting"):
            self.retained()

    def test_conflicting_resources_rejected(self):
        log = LOG.replace("0 bytes spill stores", "8 bytes spill stores")
        self.write_receipt(self.metadata, log, 1)
        with self.assertRaisesRegex(RuntimeError, "Conflicting"):
            self.retained()

    def test_stale_ptx_hash_fails_discovery(self):
        self.assert_rejected({**self.metadata, "ptx_sha256": "a" * 64})

    def test_changed_ptx_content_does_not_reuse_old_receipt(self):
        self.ptx.write_bytes(PTX + b"// changed compilation input\n")
        with patch.object(bootstrap, "ptxas_kernel_stats") as assembler:
            self.assertEqual(self.invoke(), 1)
            assembler.assert_not_called()

    def test_actual_tool_hash_mismatch(self):
        self.tool.write_bytes(b"changed tool")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.retained()

    def test_unknown_schema(self):
        self.assert_rejected({**self.metadata, "schema": 2})

    def test_unknown_metadata_field(self):
        self.assert_rejected({**self.metadata, "invented": 0})

    def test_duplicate_json_field(self):
        text = json.dumps(self.metadata)[:-1] + ',"schema":1}'
        self.receipt.write_text(text, encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "Duplicate JSON"):
            self.retained()

    def test_unknown_policy_flag(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["options"].append("--maxrregcount=32")
        self.assert_rejected(metadata)

    def test_duplicate_policy_flag(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["options"].append("--fmad=false")
        self.assert_rejected(metadata)

    def test_invalid_split_budget(self):
        for budget in ("0", "-1", "4294967296", "all"):
            with self.subTest(budget=budget):
                metadata = copy.deepcopy(self.metadata)
                metadata["options"][2] = "--split-compile=" + budget
                self.assert_rejected(metadata)

    def test_wrong_metadata_arch(self):
        self.assert_rejected({**self.metadata, "arch": "sm_89"})

    def test_wrong_options_arch(self):
        metadata = copy.deepcopy(self.metadata)
        metadata["options"][0] = "--gpu-name=sm_89"
        self.assert_rejected(metadata)

    def test_wrong_raw_arch(self):
        self.assert_rejected(log=LOG.replace("'sm_86'", "'sm_89'"))

    def test_duplicate_entries_inventory(self):
        self.assert_rejected({**self.metadata, "entries": ["kernel", "kernel"]})

    def test_missing_entries_inventory(self):
        self.assert_rejected({**self.metadata, "entries": []})

    def test_unknown_entries_inventory(self):
        self.assert_rejected({**self.metadata, "entries": ["other"]})

    def test_duplicate_helper_section(self):
        self.assert_rejected(log=LOG + "ptxas info : Function properties for helper\n")

    def test_duplicate_counter(self):
        self.assert_rejected(log=LOG + "  0 bytes spill loads\n")

    def test_missing_helper_counter(self):
        self.assert_rejected(log=LOG.rsplit(", 0 bytes spill loads", 1)[0] + "\n")

    def test_missing_entry_register_count(self):
        self.assert_rejected(log=LOG.replace("ptxas info : Used 8 registers\n", ""))

    def test_unknown_reported_function(self):
        self.assert_rejected(
            log=LOG.replace("properties for helper", "properties for unknown_helper"),
        )

    def test_missing_raw_log(self):
        (self.receipt.parent / "sm_86.ptxas.log").unlink()
        with self.assertRaises(OSError):
            self.retained()

    def test_duplicate_raw_entry(self):
        self.assert_rejected(
            log=LOG + "ptxas info : Compiling entry function 'kernel' for 'sm_86'\n",
        )

    def test_missing_raw_entry(self):
        self.assert_rejected(log=LOG.split("\n", 1)[1])

    def test_helper_only_spill_is_gated(self):
        before, after = LOG.rsplit("0 bytes spill stores", 1)
        self.write_receipt(self.metadata, before + "8 bytes spill stores" + after)
        self.assertEqual(self.invoke(), 1)

    def test_frame_distinct_from_spill(self):
        log = LOG.replace("0 bytes stack frame", "16 bytes stack frame")
        self.write_receipt(self.metadata, log)
        self.assertEqual(self.invoke(), 0)
        self.assertEqual(self.invoke(strict=True), 1)

    def test_unreported_helpers_remain_explicitly_unknown(self):
        log = LOG.split("ptxas info : Function properties for helper", 1)[0]
        self.write_receipt(self.metadata, log)
        statistics, _ = self.retained()
        report = bootstrap.kernel_resource_report(statistics, self.entries, self.calls)
        self.assertNotIn("helper", report["functions"])
        self.assertEqual(report["entries"]["kernel"]["unreported_ptx_helpers"], ["helper"])
        self.assertNotIn("helper", report["entries"]["kernel"]["reported_helpers"])

    def test_missing_receipt_never_implicitly_assembles(self):
        self.receipt.unlink()
        with patch.object(
            bootstrap, "ptxas_kernel_stats", side_effect=AssertionError("unexpected assembly"),
        ) as assembler:
            self.assertEqual(self.invoke(), 1)
            assembler.assert_not_called()

    def test_explicit_standalone_source(self):
        statistics = bootstrap.parse_ptxas_stats(LOG)
        standalone = (
            statistics, self.root / "sm_86.cubin",
            {"resource_source": "standalone-ptxas"},
        )
        with patch.object(
            bootstrap, "ptxas_kernel_stats", return_value=standalone,
        ) as assembler:
            self.assertEqual(self.invoke(assemble=True), 0)
            assembler.assert_called_once()
        report = json.loads((self.root / "kernels/sm_86-resources.json").read_text())
        self.assertEqual(report["provenance"]["resource_source"], "standalone-ptxas")

    def test_lines_requires_explicit_assembly(self):
        with patch.object(
            bootstrap, "build_env", side_effect=AssertionError("unexpected build"),
        ):
            self.assertEqual(self.invoke(lines=True), 1)

    def test_raw_log_path_cannot_escape_receipt(self):
        self.assert_rejected({**self.metadata, "raw_log": "../different.log"})

    def test_false_completion_claim(self):
        self.assert_rejected({**self.metadata, "resource_coverage_complete": True})

    @unittest.skipUnless(
        os.environ.get("WARPBRO_TEST_PTXAS_RECEIPT"),
        "set WARPBRO_TEST_PTXAS_RECEIPT to validate an existing real compiler receipt",
    )
    def test_actual_production_receipt(self):
        receipt = Path(os.environ["WARPBRO_TEST_PTXAS_RECEIPT"])
        metadata = json.loads(receipt.read_text(encoding="utf-8"))
        if explicit := os.environ.get("WARPBRO_TEST_PTX"):
            source = Path(explicit)
        else:
            source = receipt.parent / metadata["ptx_name"]
            if not source.is_file():
                source = receipt.parent.parent / metadata["ptx_name"]
        data = source.read_bytes()
        entries, calls = bootstrap.ptx_call_graph(data.decode("utf-8"))
        statistics, provenance = bootstrap.read_ptxas_receipt(
            receipt, source, metadata["arch"],
            hashlib.sha256(data.removesuffix(b"\0")).hexdigest(), entries, calls,
        )
        self.assertEqual(set(metadata["entries"]), entries)
        self.assertLessEqual(entries, statistics.keys())
        self.assertEqual(provenance["resource_source"], "production-aot-receipt")


if __name__ == "__main__":
    unittest.main()
