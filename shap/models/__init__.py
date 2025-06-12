# from ._model import Model
# from ._teacher_forcing import TeacherForcing
# from ._text_generation import TextGeneration
# from ._topk_lm import TopKLM
# from ._transformers_pipeline import TransformersPipeline
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submodules=[],
    submod_attrs={
        "_model": ["Model"],
        "_teacher_forcing": ["TeacherForcing"],
        "_text_generation": ["TextGeneration"],
        "_topk_lm": ["TopKLM"],
        "_transformers_pipeline": ["TransformersPipeline"],
    },
)

# __all__ = [
#     "Model",
#     "TeacherForcing",
#     "TextGeneration",
#     "TopKLM",
#     "TransformersPipeline",
# ]
