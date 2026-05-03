from __future__ import annotations

import faulthandler
import os
import platform
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class CrashLoggerConfig:
    """Configuration for optional crash/segfault diagnostics.

    If ``enabled`` is False, no handlers are installed.
    If ``log_path`` is provided, logs are written there; otherwise stderr is used.
    """

    enabled: bool = False
    log_path: str | None = None


def _write_runtime_header(stream: object) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    header_lines = [
        "=" * 80,
        "Crash diagnostics enabled",
        f"utc_time={ts}",
        f"pid={os.getpid()}",
        f"python={sys.version.splitlines()[0]}",
        f"platform={platform.platform()}",
        f"argv={sys.argv}",
        "=" * 80,
        "",
    ]
    for line in header_lines:
        stream.write(f"{line}\n")
    stream.flush()


def _install_signal_backtraces() -> None:
    # User-triggered tracebacks (where supported) to capture hanging/deadlock states.
    for sig_name in ("SIGUSR1", "SIGUSR2"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            faulthandler.register(sig, all_threads=True, chain=False)


def enable_crash_logging(config: CrashLoggerConfig) -> None:
    """Enable low-level crash diagnostics useful for investigating segfaults.

    This function is best-effort and intentionally avoids raising when a specific
    diagnostic hook cannot be installed on the current platform.
    """

    if not config.enabled:
        return

    target = None
    if config.log_path:
        log_file = Path(config.log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        target = log_file.open("a", encoding="utf-8")

    if target is None:
        faulthandler.enable(all_threads=True)
        _write_runtime_header(sys.stderr)
    else:
        faulthandler.enable(file=target, all_threads=True)
        _write_runtime_header(target)

    _install_signal_backtraces()
