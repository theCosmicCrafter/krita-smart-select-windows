from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT_DIR / "worker" / "smart_select_worker.py"


def load_worker_module() -> Any:
    spec = importlib.util.spec_from_file_location("smart_select_worker", WORKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load worker module from {WORKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_npy_header(path: Path) -> tuple[str, tuple[int, ...], bytes]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY\x01\x00"):
        raise AssertionError("expected NumPy v1 .npy file")

    header_length = struct.unpack("<H", data[8:10])[0]
    header = data[10 : 10 + header_length].decode("latin1")
    descr = header.split("'descr': '", 1)[1].split("'", 1)[0]
    shape_text = header.split("'shape': (", 1)[1].split(")", 1)[0]
    shape = tuple(int(value.strip()) for value in shape_text.split(",") if value.strip())
    payload = data[10 + header_length :]
    return descr, shape, payload


class SmartSelectWorkerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.worker = load_worker_module()
        self.worker.cancelled_jobs.clear()

    def test_fake_segments_returns_three_hit_testable_segments(self) -> None:
        segment_map, metadata = self.worker.fake_segments(64, 48)

        self.assertEqual(len(segment_map), 64 * 48)
        self.assertEqual([item["id"] for item in metadata], [1, 2, 3])
        self.assertTrue(all(item["area"] > 0 for item in metadata))
        self.assertEqual(segment_map[4 * 64 + 5], 1)
        self.assertEqual(segment_map[24 * 64 + 32], 2)
        self.assertEqual(segment_map[32 * 64 + 43], 3)

    def test_prepare_segments_emits_progress_result_and_uint16_npy(self) -> None:
        output = io.StringIO()
        cleanup_dir: Path | None = None
        try:
            with contextlib.redirect_stdout(output):
                self.worker.handle_prepare_segments(
                    7,
                    {
                        "image_path": "",
                        "model_id": "mock",
                        "fingerprint": "mock-no-snapshot-v1",
                        "canvas_width": 32,
                        "canvas_height": 24,
                    },
                )

            messages = [json.loads(line) for line in output.getvalue().splitlines()]
            self.assertEqual(
                [message.get("stage") for message in messages[:4]],
                [
                    "loading_model",
                    "reading_image",
                    "generating_segments",
                    "building_hover_index",
                ],
            )

            result = messages[-1]["result"]
            segment_map_path = Path(result["segment_map_path"])
            segments_path = Path(result["segments_path"])
            cleanup_dir = segment_map_path.parent
            self.assertTrue(segment_map_path.exists())
            self.assertTrue(segments_path.exists())
            self.assertEqual(len(result["metadata"]), 3)

            descr, shape, payload = parse_npy_header(segment_map_path)
            self.assertEqual(descr, "<u2")
            self.assertEqual(shape, (24, 32))
            self.assertEqual(len(payload), 24 * 32 * 2)
        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)

    def test_prepare_segments_rejects_invalid_canvas_size(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_prepare_segments(8, {"canvas_width": 0, "canvas_height": 24})

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(
            messages,
            [
                {
                    "id": 8,
                    "error": {
                        "type": "InvalidInput",
                        "message": "Canvas size is invalid",
                    },
                }
            ],
        )

    def test_handle_request_rejects_non_object_params(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_request({"id": 9, "method": "prepare_segments", "params": []})

        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0]["error"]["type"], "InvalidInput")

    def test_cancel_request_records_cancelled_job(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.worker.handle_request({"id": 10, "method": "cancel", "params": {"job_id": 7}})

        self.assertIn(7, self.worker.cancelled_jobs)
        messages = [json.loads(line) for line in output.getvalue().splitlines()]
        self.assertEqual(messages[0], {"id": 10, "result": {"cancelled": True}})

    def test_worker_process_accepts_ndjson_request(self) -> None:
        request = {
            "id": 11,
            "method": "prepare_segments",
            "params": {
                "image_path": "",
                "model_id": "mock",
                "fingerprint": "mock-no-snapshot-v1",
                "canvas_width": 16,
                "canvas_height": 12,
            },
        }

        completed = subprocess.run(
            [sys.executable, str(WORKER_PATH)],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(messages[-1]["id"], 11)
        self.assertIn("result", messages[-1])
        cleanup_dir = Path(messages[-1]["result"]["segment_map_path"]).parent
        shutil.rmtree(cleanup_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
