"""核验本轮冻结模型及交接资产的文件完整性。

逐项检查交接清单及显式输入的 SHA256，保存可追溯报告，用于后续模型加载前的资产核验。
输入：--config 指定交接目录、清单 SHA、独立文件 SHA 和报告路径。输出：JSON 完整性报告。
"""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return {"path": str(path), "resolved_path": str(path.resolve()), "sha256": actual,
            "bytes": path.stat().st_size}


def verify(config_path):
    config = json.loads(config_path.read_text())
    checked = []
    for bundle in config["bundles"]:
        root = Path(bundle["root"])
        manifest = root / "SHA256SUMS"
        checked.append(check(manifest, bundle["manifest_sha256"]))
        for line in manifest.read_text().splitlines():
            expected, relative = line.split(maxsplit=1)
            path = root / relative.removeprefix("*")
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError(f"Manifest path outside bundle: {relative}")
            checked.append(check(path, expected))
    for item in config["files"]:
        checked.append(check(Path(item["path"]), item["sha256"]))
    report = {"format": "r1lite_asset_integrity_v1", "config_sha256": sha256(config_path),
              "integrity_only": True, "files": checked}
    output = Path(config["report"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.config.resolve()))


if __name__ == "__main__":
    main()
