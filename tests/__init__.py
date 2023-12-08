from pathlib import Path


PROJECT_ROOT = Path.joinpath(Path(__file__).parent, "..").resolve()
TESTS_ROOT = Path.joinpath(PROJECT_ROOT, "tests")
FIXTURES_ROOT = Path.joinpath(PROJECT_ROOT, "storage", "fixtures")
