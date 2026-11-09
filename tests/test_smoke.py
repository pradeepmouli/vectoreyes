import vectoreyes


def test_version_exposed():
    assert isinstance(vectoreyes.__version__, str)
    assert vectoreyes.__version__.count(".") == 2
