import json

import job_status


def test_a_successful_run_is_recorded_without_an_error(tmp_path):
    path = str(tmp_path / "docs" / "job_status.json")
    run = job_status.record_run(path, "cycle", "2026-09-27T14:45:01+00:00", 0, "cycle", "lots of output")
    assert run["status"] == "ok" and run["error"] == ""
    data = json.loads(open(path, encoding="utf-8").read())
    assert data["runs"] == [run] and data["generated_at"]


def test_a_failed_run_keeps_the_stage_and_last_output_line(tmp_path):
    path = str(tmp_path / "job_status.json")
    log = "collecting...\nFAILED tests/test_x.py::test_y\n==== 2 failed, 400 passed in 5.1s ====\n\n"
    run = job_status.record_run(path, "cycle", "2026-09-27T14:45:01+00:00", 1, "pytest", log)
    assert run["status"] == "failed"
    assert run["stage"] == "pytest"
    assert run["error"] == "2 failed, 400 passed in 5.1s"


def test_the_error_line_is_sanitized_and_truncated(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPACA_API_SECRET", "super-secret-value-123")
    path = str(tmp_path / "job_status.json")
    log = "Traceback...\nRuntimeError: auth failed with super-secret-value-123 " + "x" * 500
    run = job_status.record_run(path, "cycle", "t", 1, "cycle", log)
    assert "super-secret-value-123" not in run["error"]
    assert len(run["error"]) <= job_status.MAX_ERROR_CHARS


def test_history_is_capped_and_survives_a_broken_file(tmp_path):
    path = tmp_path / "job_status.json"
    path.write_text("not json", encoding="utf-8")
    for i in range(job_status.MAX_RUNS + 5):
        job_status.record_run(str(path), "cycle", f"2026-09-27T00:{i:02d}:00+00:00", 0, "cycle")
    runs = json.loads(path.read_text(encoding="utf-8"))["runs"]
    assert len(runs) == job_status.MAX_RUNS
    assert runs[-1]["started_at"].startswith("2026-09-27T00:64")


def test_cli_reads_the_log_file(tmp_path):
    log = tmp_path / "run.log"
    log.write_text("ValueError: boom\n", encoding="utf-8")
    path = tmp_path / "job_status.json"
    assert job_status.main(["record", "--job", "cycle", "--started", "s", "--rc", "2",
                            "--stage", "cycle", "--log", str(log), "--path", str(path)]) == 0
    run = json.loads(path.read_text(encoding="utf-8"))["runs"][0]
    assert run == {**run, "status": "failed", "stage": "cycle", "error": "ValueError: boom"}


def test_run_job_records_status_and_returns_the_real_exit_code():
    text = open("deploy/run_job.sh", encoding="utf-8").read()
    assert "job_status.py record --job cycle" in text
    assert "docs/job_status.json" in text
    assert 'exit "$JOB_RC"' in text
