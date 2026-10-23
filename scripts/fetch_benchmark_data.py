#!/usr/bin/env python3
"""Stage the two public LIBSVM inputs and record their exact content hashes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    records = []
    for name in ("a9a", "w8a"):
        url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/" + name
        target = args.directory / name
        if not target.exists():
            temporary = args.directory / (name + ".download")
            with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
                while chunk := response.read(1 << 20):
                    output.write(chunk)
            temporary.replace(target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        records.append(dict(file=name, url=url, bytes=target.stat().st_size, sha256=digest))
    manifest = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), files=records)
    (args.directory / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
