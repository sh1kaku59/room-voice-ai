import asyncio
import inspect
import sys
from pathlib import Path

# Add room_voice_ai to sys.path
ROOT_DIR = Path(__file__).resolve().parent / "room_voice_ai"
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_pyfunc_call(pyfuncitem):
    """Allow running async def tests natively without pytest-asyncio plugin."""
    testfunction = pyfuncitem.obj
    if inspect.iscoroutinefunction(testfunction):
        argnames = pyfuncitem._fixtureinfo.argnames
        funcargs = pyfuncitem.funcargs
        testargs = {arg: funcargs[arg] for arg in argnames}
        asyncio.run(testfunction(**testargs))
        return True
    return None

