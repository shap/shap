import pytest
import time

def load_tokenizer_model(name: str, retries: int) -> tuple["transformers.PreTrainedTokenizer", "transformers.PreTrainedModel"]:
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    AutoModelForSeq2SeqLM = pytest.importorskip("transformers").AutoModelForSeq2SeqLM

    max_retries = retries  # Use the parameter value
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            return tokenizer, model
        except OSError:
            time.sleep(2 ** attempt)  # Exponential backoff
    raise OSError(f"Failed to load model and tokenizer after {max_retries} attempts")


@pytest.fixture(scope="session")
def basic_translation_scenario():
    """Create a basic transformers translation model and tokenizer."""
    # Use a *tiny* tokenizer model, to keep tests running as fast as possible.
    # Nb. At time of writing, this pretrained model requires "protobuf==3.20.3".
    # name = "mesolitica/finetune-translation-t5-super-super-tiny-standard-bahasa-cased"
    # name = "Helsinki-NLP/opus-mt-en-es"
    name = "hf-internal-testing/tiny-random-BartModel"

    tokenizer, model = load_tokenizer_model(name=name, retries=5)

    # define the input sentences we want to translate
    data = [
        "In this picture, there are four persons: my father, my mother, my brother and my sister.",
        "Transformers have rapidly become the model of choice for NLP problems, replacing older recurrent neural network models",
    ]

    return model, tokenizer, data
