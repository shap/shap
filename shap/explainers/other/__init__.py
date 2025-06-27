import lazy_loader as lazy

# Use lazy.attach_stub to enable proper type checking for other explainers
__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
