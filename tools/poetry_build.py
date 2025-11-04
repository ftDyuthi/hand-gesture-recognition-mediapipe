#!/usr/bin/env python3
"""
Poetry build helper: run inside the poetry-managed venv as

  poetry run build -- --mode onedir

(we add a -- because Poetry passes args after the script name)

This script will:
- discover OpenCV binaries in the active environment and add them to the PyInstaller command
- run PyInstaller (onedir or onefile) using the active Python interpreter
- produce the build in ./dist and print a SHA256
"""
import argparse
import os
import site
import shutil
import subprocess
import sys
from pathlib import Path


def find_cv2_binaries():
    """Return a list of (src, dest) pairs for opencv binaries to include with --add-binary"""
    pairs = []
    # Try typical locations: site-packages/cv2 and site-packages/opencv_python.libs
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        sp_path = Path(sp)
        if not sp_path.exists():
            continue
        cv2_dir = sp_path / "cv2"
        if cv2_dir.exists():
            # find main .pyd
            for p in cv2_dir.glob("*.pyd"):
                pairs.append((str(p), "."))
            # also include any dlls in cv2 folder
            for p in cv2_dir.glob("*.dll"):
                pairs.append((str(p), "."))
        libs_dir = sp_path / "opencv_python.libs"
        if libs_dir.exists():
            # include all dlls in a folder inside the dist
            dlls = list(libs_dir.glob("*.dll"))
            if dlls:
                # use wildcard include by pointing at the directory; PyInstaller doesn't accept directory wildcard, so include each
                for p in dlls:
                    pairs.append((str(p), "opencv_python.libs"))
    # Deduplicate
    uniq = []
    seen = set()
    for s, d in pairs:
        key = (os.path.normpath(s), d)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((s, d))
    return uniq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("onedir", "onefile"), default="onedir")
    parser.add_argument("--name", default="hand_gesture")
    parser.add_argument("--no-upx", action="store_true")
    args, unknown = parser.parse_known_args()

    py = sys.executable
    print(f"Using Python: {py}")

    # Discover cv2 binaries
    print("Discovering OpenCV binaries in site-packages...")
    cv2_pairs = find_cv2_binaries()
    if not cv2_pairs:
        print("No OpenCV binaries discovered automatically. Ensure opencv is installed in the env or supply binaries manually.")
    else:
        print(f"Found {len(cv2_pairs)} OpenCV binary files to include.")

    # Build PyInstaller command
    cmd = [py, "-m", "PyInstaller", "--noconfirm", "--clean"]
    if args.mode == "onefile":
        cmd.append("--onefile")
    else:
        cmd.append("--onedir")
    if args.no_upx or args.no_upx:
        cmd.append("--noupx")
    cmd += ["--name", args.name, "--distpath", "dist", "--workpath", "build"]

    # add binaries
    for src, dest in cv2_pairs:
        spec = f"{src};{dest}"
        cmd += ["--add-binary", spec]

    # add data dirs
    cmd += ["--add-data", "model;model", "--add-data", "utils;utils"]

    # append app.py
    cmd.append("app.py")

    print("Running PyInstaller with command:")
    print(" ".join(cmd))

    # Run command
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"PyInstaller failed with exit code {res.returncode}")
        sys.exit(res.returncode)

    # Compute checksum for produced artifact
    if args.mode == "onefile":
        artifact = Path("dist") / f"{args.name}.exe"
    else:
        artifact = Path("dist") / args.name / f"{args.name}.exe"
    if artifact.exists():
        import hashlib

        h = hashlib.sha256()
        with artifact.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        print("Built:", artifact)
        print("SHA256:", h.hexdigest())
    else:
        print("No artifact found at expected path:", artifact)
        sys.exit(2)


if __name__ == "__main__":
    main()
