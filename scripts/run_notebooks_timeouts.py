import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

TIMEOUT = 20  # seconds

allow_to_fail = [
    Path("tabular_examples/tree_based_models/tree_shap_paper/Tree SHAP in Python.ipynb"),
    Path("api_examples/plots/decision_plot.ipynb"),
    Path("image_examples/image_classification/Image Multi Class.ipynb"),
    Path("tabular_examples/model_agnostic/Multioutput Regression SHAP.ipynb"),
    Path("tabular_examples/neural_networks/Census income classification with Keras.ipynb"),
    Path("api_examples/explainers/GPUTree.ipynb"),
    Path("genomic_examples/DeepExplainer Genomics Example.ipynb"),
    Path("tabular_examples/tree_based_models/Perfomance Comparison.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Figure 7 - Airline Tweet Sentiment Analysis.ipynb"),
    Path("overviews/Be careful when interpreting predictive models in search of causal insights.ipynb"),
    Path("benchmarks/text/Text Emotion Multiclass Classification Benchmark Demo.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Figure 6 - Supervised Clustering R-squared.ipynb"),
    Path("overviews/Explaining quantitative measures of fairness.ipynb"),
    Path("image_examples/image_captioning/Image Captioning using Open Source.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Performance comparison.ipynb"),
    Path("tabular_examples/tree_based_models/League of Legends Win Prediction with XGBoost.ipynb"),
    Path("overviews/An introduction to explainable AI with Shapley values.ipynb"),
    Path("image_examples/image_captioning/Image Captioning using Azure Cognitive Services.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Figures 8-11 NHANES I Survival Model.ipynb"),
    Path("benchmarks/others/Benchmark Debug Mode.ipynb"),
    Path("tabular_examples/tree_based_models/NHANES I Survival Model.ipynb"),
    Path("benchmarks/text/Abstractive Summarization Benchmark Demo.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Performance comparison copy.ipynb"),
    Path("tabular_examples/tree_based_models/tree_shap_paper/Figures 8-11 NHANES I Survival Model-Copy1.ipynb"),
]

allow_to_timeout = [
    Path("tabular_examples/model_agnostic/Census income classification with scikit-learn.ipynb"),
    Path("image_examples/image_classification/Explain MobilenetV2 using the Partition explainer (PyTorch).ipynb"),
    Path("text_examples/sentiment_analysis/Emotion classification multiclass example.ipynb"),
    Path("api_examples/plots/bar.ipynb"),
    Path("text_examples/language_modelling/Language Modeling Explanation Demo.ipynb"),
    Path("image_examples/image_classification/Explain an Intermediate Layer of VGG16 on ImageNet (PyTorch).ipynb"),
    Path("text_examples/sentiment_analysis/Keras LSTM for IMDB Sentiment Classification.ipynb"),
    Path("text_examples/sentiment_analysis/Positive vs. Negative Sentiment Classification.ipynb"),
    Path("image_examples/image_classification/Multi-class ResNet50 on ImageNet (TensorFlow).ipynb"),
    Path("tabular_examples/tree_based_models/Fitting a Linear Simulation with XGBoost.ipynb"),
    Path("text_examples/question_answering/Explaining a Question Answering Transformers Model.ipynb"),
    Path("text_examples/sentiment_analysis/Using custom functions and tokenizers.ipynb"),
    Path("text_examples/translation/Machine Translation Explanations.ipynb"),
    Path("image_examples/image_classification/Explain an Intermediate Layer of VGG16 on ImageNet.ipynb"),
    Path("image_examples/image_classification/Multi-class ResNet50 on ImageNet (TensorFlow)-checkpoint.ipynb"),
    Path("api_examples/plots/text.ipynb"),
    Path("image_examples/image_classification/Explain ResNet50 using the Partition explainer.ipynb"),
    Path("benchmarks/tabular/Tabular Prediction Benchmark Demo.ipynb"),
    Path("text_examples/text_generation/Open Ended GPT2 Text Generation Explanations.ipynb"),
    Path("image_examples/image_classification/Front Page DeepExplainer MNIST Example.ipynb"),
    Path("tabular_examples/tree_based_models/Census income classification with XGBoost.ipynb"),
    Path("api_examples/plots/waterfall.ipynb"),
    Path("benchmarks/text/Machine Translation Benchmark Demo.ipynb"),
    Path("text_examples/summarization/Abstractive Summarization Explanation Demo.ipynb"),
    Path("api_examples/plots/beeswarm.ipynb"),
    Path("api_examples/plots/image.ipynb"),
    Path("benchmarks/tabular/Benchmark XGBoost explanations.ipynb"),
    Path("benchmarks/image/Image Multiclass Classification Benchmark Demo.ipynb"),
    Path("api_examples/plots/scatter.ipynb"),
    Path("text_examples/text_entailment/Textual Entailment Explanation Demo.ipynb"),
    Path("image_examples/image_classification/Multi-input Gradient Explainer MNIST Example.ipynb"),
    Path("image_examples/image_classification/PyTorch Deep Explainer MNIST example.ipynb"),
]


def main():
    notebooks_directory = Path('notebooks')
    error_notebooks = []
    ep = ExecutePreprocessor(timeout=TIMEOUT, log_level=40)

    error_notebooks = []
    timeout_notebooks = []
    notebooks_to_run = set(notebooks_directory.rglob('*.ipynb'))\
        - set([notebooks_directory / nb for nb in allow_to_fail])\
        - set([notebooks_directory / nb for nb in allow_to_timeout])
    for notebook_path in notebooks_to_run:
        with open(notebook_path) as f:
            nb = nbformat.read(f, as_version=4)
        start_time = time.time()
        try:
            ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
            print(f"Executed notebook {notebook_path} in {time.time() - start_time:.2f} seconds.")
        except CellExecutionError:
            error_notebooks.append(notebook_path)
        except TimeoutError:
            print(f"Execution of {notebook_path} timed out after {TIMEOUT} seconds.")
            timeout_notebooks.append(notebook_path)

    if len(error_notebooks) > 0:
        print(f"Notebooks failed with error codes: {', '.join([str(nb) for nb in error_notebooks])}")
    if len(timeout_notebooks) > 0:
        print(f"Notebooks timed out: {', '.join([str(nb) for nb in timeout_notebooks])}")
    if len(error_notebooks) > 0 or len(timeout_notebooks) > 0:
        raise Exception("Notebooks failed to execute.")
    else:
        print("All notebooks executed successfully.")

if __name__ == "__main__":
    main()
