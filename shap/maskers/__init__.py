import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_composite": ["Composite"],
        "_fixed": ["Fixed"],
        "_fixed_composite": ["FixedComposite"],
        "_image": ["Image"],
        "_masker": ["Masker"],
        "_output_composite": ["OutputComposite"],
        "_tabular": ["Impute", "Independent", "Partition"],
        "_text": ["Text"],
    },
)
