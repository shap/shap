import time

import pytest


def load_tokenizer_model(name: str, retries: int) -> tuple:
    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer
    AutoModelForSeq2SeqLM = pytest.importorskip("transformers").AutoModelForSeq2SeqLM

    max_retries = retries  # Use the parameter value
    for attempt in range(max_retries):
        try:
            tokenizer = AutoTokenizer.from_pretrained(name)
            model = AutoModelForSeq2SeqLM.from_pretrained(name)
            return tokenizer, model
        except OSError:
            time.sleep(2**attempt)  # Exponential backoff
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


@pytest.fixture()
def causalml_synth_data():
    """
    Generates synthetic data for causalml causal tree tests.

    Unlike standard regression trees causal trees in causalml evaluate outcome conditioning on treatments
    Thus, a causal tree estimates Y_hat|X,T=t, where t={0, 1,..., n}.
    The simplest case is when T = {0, 1}, 0 - no treatment, 1 - some treatment.
    """
    dataset = pytest.importorskip("causalml.dataset")

    data_mode = 1  # Basic synthetic data mode with a difficult nuisance components and an easy treatment effect
    sigma = 0.1  # Synthetic standard deviation of the error term
    n_observations = 100  # The number of samples to generate
    n_features = 8  # X in (Y_hat|X, T=0, Y_hat|X, T=1)
    n_outcomes = 2  # Treatment conditioned outcomes: (Y_hat|X,T=0, Y_hat|X,T=1)

    data = dataset.synthetic_data(mode=data_mode, n=n_observations, p=n_features, sigma=sigma)
    return data, n_outcomes
