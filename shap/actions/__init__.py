import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_action": ["Action"],
    },
)
# from ._action import Action
#
# __all__ = ["Action"]
