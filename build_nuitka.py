"""Build app.py into a single-file Windows .exe with Nuitka.

Usage (from project root, with .venv activated or via its python):
    .venv/Scripts/python build_nuitka.py

Outputs:
    dist/app.exe
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ENTRY = ROOT / "app.py"
ICON = Path(r"C:\Users\euler\Downloads\icon_v2_lambda.png")
DIST = ROOT / "dist"


def main() -> int:
    if not ENTRY.exists():
        print(f"error: entry script not found: {ENTRY}", file=sys.stderr)
        return 1
    if not ICON.exists():
        print(f"warning: icon not found at {ICON}; building without icon", file=sys.stderr)
        icon_args: list[str] = []
    else:
        # Nuitka accepts PNG here and converts to ICO if Pillow is available.
        icon_args = [f"--windows-icon-from-ico={ICON}"]

    DIST.mkdir(exist_ok=True)

    # Locate the pythonnet/runtime folder of the active Python — its
    # Python.Runtime.dll must travel with the bundle or pywebview will
    # raise "You must have pythonnet installed".
    try:
        import pythonnet  # type: ignore
        pythonnet_runtime = Path(pythonnet.__file__).parent / "runtime"
    except Exception:
        pythonnet_runtime = None
    pythonnet_args: list[str] = []
    if pythonnet_runtime and pythonnet_runtime.exists():
        pythonnet_args.append(
            f"--include-data-dir={pythonnet_runtime}=pythonnet/runtime"
        )
    else:
        print("warning: could not locate pythonnet/runtime; bundle may fail at startup",
              file=sys.stderr)

    cmd = [
        sys.executable, "-m", "nuitka",
        "--onefile",
        "--assume-yes-for-downloads",
        "--windows-console-mode=disable",
        # Fixed extraction path so a single Defender exclusion covers it.
        # Default %TEMP%\onefile_<pid>_<random>\ has random names, making
        # exclusion impractical and triggering AV on the extracted app.dll.
        "--onefile-tempdir-spec={CACHE_DIR}/GeneralizedCovarianceMatrix/runtime",
        # numpy/scipy plugins handle bundling; the pywebview Nuitka plugin
        # auto-loads and manages webview.* — do NOT also pass
        # --include-package=webview (causes plugin/user decision conflict).
        "--include-package=rmt_denoise",
        "--include-package=plotly",
        "--include-package=tqdm",
        # pywebview's Edge/WebView2 backend on Windows is loaded via pythonnet
        # at runtime, so Nuitka cannot infer the dependency — include explicitly.
        "--include-package=pythonnet",
        "--include-package=clr_loader",
        *pythonnet_args,
        "--nofollow-import-to=webview.platforms.android",
        "--nofollow-import-to=webview.platforms.cocoa",
        "--nofollow-import-to=webview.platforms.gtk",
        "--nofollow-import-to=webview.platforms.qt",
        f"--output-dir={DIST}",
        "--output-filename=GeneralizedCovarianceMatrix.exe",
        "--remove-output",
        "--company-name=random_matrix_ESD",
        "--product-name=Random Matrix ESD",
        "--file-version=1.0.0.0",
        "--product-version=1.0.0.0",
        *icon_args,
        str(ENTRY),
    ]

    print(">> running:", " ".join(cmd))
    try:
        result = subprocess.run(cmd, cwd=ROOT)
    except FileNotFoundError:
        print("error: Nuitka not installed. Run: pip install nuitka", file=sys.stderr)
        return 1

    if result.returncode != 0:
        print(f"\nNuitka failed with exit code {result.returncode}", file=sys.stderr)
        return result.returncode

    exe = DIST / "GeneralizedCovarianceMatrix.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"\nbuilt: {exe}  ({size_mb:.1f} MB)")
    else:
        print(f"\nbuild finished but {exe} not found", file=sys.stderr)
        return 1

    # Bump exe stack reserve. .NET CLR initialization (loaded via pythonnet
    # for pywebview's WinForms backend) recurses deep enough to overflow the
    # default 1 MB Windows thread stack, crashing webview.start() with
    # STATUS_STACK_OVERFLOW (0xC00000FD).
    editbin = next(
        Path(r"C:\Program Files\Microsoft Visual Studio").rglob(
            r"Hostx64\x64\editbin.exe"
        ),
        None,
    ) or next(
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio").rglob(
            r"Hostx64\x64\editbin.exe"
        ),
        None,
    )
    if editbin:
        print(f">> bumping stack reserve via {editbin}")
        subprocess.run(
            [str(editbin), "/STACK:33554432,1048576", str(exe)], check=False
        )
    else:
        print("warning: editbin.exe not found; stack-reserve patch skipped — "
              "exe may crash with STATUS_STACK_OVERFLOW.", file=sys.stderr)

    # Tidy: nuitka leaves <name>.build/ and <name>.dist/ next to the entry when
    # --remove-output is set this should already be cleaned, but double-check.
    for stale in (ROOT / "app.build", ROOT / "app.dist", ROOT / "app.onefile-build"):
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
