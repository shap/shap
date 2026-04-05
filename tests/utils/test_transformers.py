from shap.utils.transformers import getattr_silent


def test_getattr_silent_existing_attr():
    class Obj:
        x = 5

    assert getattr_silent(Obj(), "x") == 5


def test_getattr_silent_missing_attr():
    class Obj:
        pass

    assert getattr_silent(Obj(), "missing") is None


def test_getattr_silent_none_string_bug():
    class Obj:
        x = "None"

    assert getattr_silent(Obj(), "x") is None


def test_getattr_silent_verbose_reset():
    class Obj:
        verbose = True
        x = 42

    obj = Obj()
    getattr_silent(obj, "x")
    assert obj.verbose
