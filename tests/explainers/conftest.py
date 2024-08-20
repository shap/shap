import pytest


@pytest.fixture(scope="session")
def basic_translation_scenario():
    """Create a basic transformers translation model and tokenizer."""
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    AutoModelForSeq2SeqLM = pytest.importorskip("transformers").AutoModelForSeq2SeqLM

    # Use a *tiny* tokenizer model, to keep tests running as fast as possible.
    # Nb. At time of writing, this pretrained model requires "protobuf==3.20.3".
    # name = "mesolitica/finetune-translation-t5-super-super-tiny-standard-bahasa-cased"
    # name = "Helsinki-NLP/opus-mt-en-es"
    name = "hf-internal-testing/tiny-random-BartModel"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name)

    # define the input sentences we want to translate
    data = [
        "In this picture, there are four persons: my father, my mother, my brother and my sister.",
        "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models",
    ]

    return model, tokenizer, data
