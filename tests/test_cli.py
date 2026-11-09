import subprocess, sys
import numpy as np
from PIL import Image


def test_cli_writes_svg(tmp_path):
    img = np.full((40, 40, 3), 255, np.uint8)
    img[8:32, 8:32] = (6, 35, 54)
    src = tmp_path / "in.png"; out = tmp_path / "out.svg"
    Image.fromarray(img).save(src)
    r = subprocess.run([sys.executable, "-m", "vectoreyes.cli", str(src), "-o", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text().startswith("<svg")
