"""Compile the engine into wheels; keep editable Python sources observable."""

from pathlib import Path
from tempfile import TemporaryDirectory
import os
import shutil

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):
    def initialize(self, version, build_data):
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension
        from setuptools.command.build_ext import build_ext

        root = Path(self.root)
        source_root = root / "src"
        package_root = source_root / "vectormark"
        sources = sorted(package_root.rglob("*.pyx"))
        if version != "editable":
            sources.extend(
                path for path in sorted(package_root.rglob("*.py"))
                  if path.name not in {"__init__.py", "__main__.py", "mcp_server.py", "mcp_image.py", "cli.py"}
                  # Evaluation is Pydantic-backed orchestration, not a numeric
                  # engine kernel. Cython methods are not Python descriptors
                  # recognized by Pydantic's model namespace inspection.
                  and "evaluation" not in path.relative_to(package_root).parts
            )
        compile_args = ["-O3", "-ffp-contract=off"] if os.name != "nt" else ["/O2", "/fp:precise"]
        with TemporaryDirectory(prefix="vectormark-native-") as temporary:
            extensions = cythonize(
                [Extension(".".join(path.relative_to(source_root).with_suffix("").parts),
                           [str(path)], extra_compile_args=compile_args) for path in sources],
                build_dir=str(Path(temporary) / "cython"),
                compiler_directives={"language_level": 3, "embedsignature": True, "binding": True},
            )
            distribution = Distribution({"name": "vectormark", "ext_modules": extensions})
            command = build_ext(distribution)
            command.ensure_finalized()
            command.parallel = min(4, os.cpu_count() or 1)
            command.build_temp = str(Path(temporary) / "objects")
            # Editable imports resolve the source package, so its extension
            # must live beside the Python modules rather than in site-packages.
            command.build_lib = str(root / "src" if version == "editable" else Path(temporary) / "lib")
            command.run()
            if version != "editable":
                # The temporary directory must outlive wheel assembly.
                for extension in extensions:
                    output = Path(command.get_ext_fullpath(extension.name))
                    relative = Path(*extension.name.split(".")[:-1]) / output.name
                    destination = root / "build" / "native" / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(output, destination)
                    build_data["force_include"][str(destination)] = relative.as_posix()
        build_data["pure_python"] = False
        build_data["infer_tag"] = True
