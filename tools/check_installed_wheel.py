"""Install a wheel in an isolated package environment and exercise its assets.

Dependencies are inherited from the build environment. The application wheel
itself must load from the new environment, without editable source fallbacks.
"""
import argparse
import os
from pathlib import Path
import site
import subprocess
import sys
import tempfile
import venv


PROBE = '''
from pathlib import Path
import sys
import numpy as np
import raw_alchemy

def main():
    prefix = Path(sys.prefix).resolve()
    assert Path(raw_alchemy.__file__).resolve().is_relative_to(prefix)
    from raw_alchemy import lensfun_wrapper
    assert lensfun_wrapper._lensfun is not None, "Packaged Lensfun did not load"
    assert Path(lensfun_wrapper._lensfun._name).resolve().is_relative_to(prefix)
    database = lensfun_wrapper.LensfunDatabase()
    assert database.db, "Packaged camera database did not load"
    from raw_alchemy.pipeline.stage_identity import denoise_tag
    assert denoise_tag() is not None, "Packaged stage assets cannot be identified"
    from raw_alchemy.onnx import rgb_denoiser
    session = rgb_denoiser._get_session()
    try:
        result = session.run(None, {
            "rgb": np.full((1, 3, 512, 512), 0.2, np.float32),
            "sigma": np.full((1, 1, 512, 512), 0.25, np.float32),
        })[0]
        assert result.shape == (1, 3, 512, 512)
        assert np.isfinite(result).all()
    finally:
        session.close()
        rgb_denoiser.clear_session()
    from PySide6.QtWidgets import QApplication
    app = QApplication([])
    assert app.platformName() == "offscreen"
    print("Installed wheel: packaged Lensfun, database, model, isolated CPU inference and Qt passed")

if __name__ == "__main__":
    main()
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('wheel', type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='rawalchemy-wheel-') as work:
        root = Path(work)
        environment = root / 'environment'
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
        # A nested venv's system site is the base interpreter, not the invoking
        # build venv. Append only its installed dependency directories; do not
        # execute editable .pth files or put the source checkout on sys.path.
        package_dir = Path(subprocess.check_output(
            [str(python), '-I', '-c', 'import site; print(site.getsitepackages()[0])'],
            text=True,
        ).strip())
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / '_build_dependencies.pth').write_text(
            '\n'.join(site.getsitepackages()) + '\n', encoding='utf-8',
        )
        subprocess.run([str(python), '-m', 'pip', 'install', '--no-deps',
                        '--force-reinstall', str(args.wheel.resolve())], check=True)
        probe = root / 'probe.py'
        probe.write_text(PROBE, encoding='utf-8')
        env = os.environ.copy()
        env.pop('PYTHONPATH', None)
        env.pop('RAWALCHEMY_LENSFUN_DIR', None)
        env.update(QT_QPA_PLATFORM='offscreen', RAW_ALCHEMY_CPU_ONLY='1',
                   RAWALCHEMY_NATIVE_ISOLATION='1',
                   RAW_ALCHEMY_LOG_DIR=str(root / 'logs'))
        subprocess.run([str(python), '-I', str(probe)], cwd=root, env=env,
                       check=True, timeout=180)


if __name__ == '__main__':
    main()