"""Build both host adapters from vectorEyes's canonical packaged skills."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tomllib
from urllib.parse import urlsplit
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = "bundle-files.json"


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()


def bundle_files(endpoint: str | None) -> dict[str, bytes]:
    """Generate host configuration; never install runtimes or copy credentials."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    manifest = json.loads((ROOT / "integrations/plugin.json").read_text())
    manifest["version"] = project["version"]
    identity = {key: value for key, value in manifest.items() if key not in {"$schema", "extensions"}}
    if endpoint:
        parsed = urlsplit(endpoint)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment
                or (parsed.scheme == "http" and parsed.hostname not in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError("Use HTTPS, or loopback HTTP; endpoint must not contain credentials, query, or fragment")
        portable = {"type": "streamable-http", "url": endpoint}
        compatibility = {"type": "http", "url": endpoint}
    else:
        portable = {"type": "stdio", "command": "vectoreyes-mcp", "args": [],
                    "env": {"VECTOREYES_MCP_TRANSPORT": "stdio"}}
        compatibility = dict(portable)
    files = {
        "plugin.json": json_bytes(manifest),
        ".claude-plugin/plugin.json": json_bytes({**identity, "skills": "./skills/", "mcpServers": "./.mcp.json"}),
        ".codex-plugin/plugin.json": json_bytes({**identity, "skills": "./skills/", "mcpServers": "./.mcp.json",
                                                 "interface": manifest["extensions"]["com.openai"]["interface"]}),
        "mcp.json": json_bytes({"$schema": "https://agent-plugins.org/schemas/1.0.0/mcp.schema.json",
                                 "mcpServers": {"vectoreyes": portable}}),
        ".mcp.json": json_bytes({"mcpServers": {"vectoreyes": compatibility}}),
        "README.md": (ROOT / "docs/plugins.md").read_bytes(),
        "LICENSE": (ROOT / "LICENSE").read_bytes(),
    }
    skills = ROOT / "src/vectoreyes/assets/skills"
    for path in sorted(skills.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Plugin resources must not be symlinks: {path}")
        if path.is_file():
            files["skills/" + path.relative_to(skills).as_posix()] = path.read_bytes()
    files[INVENTORY] = json_bytes({"generated_by": "scripts/build_agent_plugin.py",
                                  "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}})
    return files


def build(output: Path, files: dict[str, bytes], *, check: bool) -> None:
    if check:
        actual = {p.relative_to(output).as_posix(): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        different = sorted(name for name in files.keys() | actual.keys() if files.get(name) != actual.get(name))
        if different:
            raise ValueError("Plugin bundle is stale: " + ", ".join(different))
        return
    if output.exists() and any(output.iterdir()) and not (output / INVENTORY).is_file():
        raise ValueError("Refusing to overwrite a non-generated directory")
    if (output / INVENTORY).exists():
        previous = json.loads((output / INVENTORY).read_text())["files"]
        for name in previous.keys() - files.keys():
            target = (output / name).resolve()
            if not target.is_relative_to(output.resolve()):
                raise ValueError("Invalid generated inventory path")
            target.unlink(missing_ok=True)
    for name, data in files.items():
        target = output / name
        if not target.resolve().is_relative_to(output.resolve()):
            raise ValueError("Output path escapes plugin directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "integrations/plugin")
    parser.add_argument("--endpoint", help="Existing HTTP MCP endpoint; omit for installed stdio runtime")
    parser.add_argument("--check", action="store_true", help="Fail on drift without changing files")
    parser.add_argument("--archive", type=Path, help="Write a deterministic, self-contained plugin ZIP")
    args = parser.parse_args()
    try:
        files = bundle_files(args.endpoint)
        build(args.output, files, check=args.check)
        if args.archive:
            if args.check:
                raise ValueError("--check does not write archives")
            args.archive.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(args.archive, "w", compression=ZIP_DEFLATED) as archive:
                for name, data in sorted(files.items()):
                    entry = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                    entry.compress_type = ZIP_DEFLATED
                    entry.external_attr = 0o100644 << 16
                    archive.writestr(entry, data)
        print(args.output)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
