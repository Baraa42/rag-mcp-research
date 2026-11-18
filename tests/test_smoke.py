"""Check that the installed package is importable."""

from importlib import import_module


def test_package_import() -> None:
    package = import_module("research_assistant")
    assert package.__name__ == "research_assistant"
