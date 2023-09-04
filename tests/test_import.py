import pkgutil
from importlib import import_module
from pathlib import Path

import ream


def test_import(pkg=ream):
    stack = [(pkg.__name__, Path(pkg.__file__).parent.absolute())]

    while len(stack) > 0:
        pkgname, pkgpath = stack.pop()
        for m in pkgutil.iter_modules([str(pkgpath)]):
            mname = f"{pkgname}.{m.name}"
            if m.ispkg:
                stack.append((mname, pkgpath / m.name))
            import_module(mname)
