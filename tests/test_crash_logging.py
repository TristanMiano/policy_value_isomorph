from policy_value_isomorph.crash_logging import CrashLoggerConfig, enable_crash_logging


def test_enable_crash_logging_writes_header(tmp_path) -> None:
    log_path = tmp_path / "crash.log"
    enable_crash_logging(CrashLoggerConfig(enabled=True, log_path=str(log_path)))
    text = log_path.read_text(encoding="utf-8")
    assert "Crash diagnostics enabled" in text
    assert "python=" in text
