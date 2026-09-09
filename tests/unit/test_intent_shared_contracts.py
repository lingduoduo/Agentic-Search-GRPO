"""The vocabulary can be consumed without importing serving or ML code."""

import subprocess
import sys
import textwrap
from pathlib import Path


def test_shared_intent_contracts_import_without_runtime_dependencies():
    source_root = str(Path(__file__).resolve().parents[2] / "src")
    script = textwrap.dedent(
        f"""
        import json
        import sys
        sys.path.insert(0, {source_root!r})

        class BlockRuntime:
            def find_spec(self, fullname, path=None, target=None):
                if (fullname.split('.')[0] in {{'numpy', 'torch', 'sentence_transformers'}}
                    or fullname.startswith(('src.model', 'src.internal.servers.web'))):
                    raise ImportError('runtime dependency: ' + fullname)
                return None

        sys.meta_path.insert(0, BlockRuntime())
        from shared_configs.intent import RouteStrategy
        assert json.dumps(RouteStrategy('search')) == '"search"'
        try:
            RouteStrategy('unknown')
        except ValueError:
            pass
        else:
            raise AssertionError('unsupported route was accepted')
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=20
    )
    assert completed.returncode == 0, completed.stderr
