import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

TIMEOUT = 20  # seconds

allow_to_fail = [
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Tree SHAP in Python.ipynb"),
    Path("notebooks/api_examples/plots/decision_plot.ipynb"),
    Path("notebooks/image_examples/image_classification/Image Multi Class.ipynb"),
    Path("notebooks/tabular_examples/model_agnostic/Multioutput Regression SHAP.ipynb"),
    Path("notebooks/tabular_examples/neural_networks/Census income classification with Keras.ipynb"),
    Path("notebooks/api_examples/explainers/GPUTree.ipynb"),
    Path("notebooks/genomic_examples/DeepExplainer Genomics Example.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/Perfomance Comparison.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Figure 7 - Airline Tweet Sentiment Analysis.ipynb"),
    Path("notebooks/overviews/Be careful when interpreting predictive models in search of causal insights.ipynb"),
    Path("notebooks/benchmarks/text/Text Emotion Multiclass Classification Benchmark Demo.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Figure 6 - Supervised Clustering R-squared.ipynb"),
    Path("notebooks/overviews/Explaining quantitative measures of fairness.ipynb"),
    Path("notebooks/image_examples/image_captioning/Image Captioning using Open Source.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Performance comparison.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/League of Legends Win Prediction with XGBoost.ipynb"),
    Path("notebooks/overviews/An introduction to explainable AI with Shapley values.ipynb"),
    Path("notebooks/image_examples/image_captioning/Image Captioning using Azure Cognitive Services.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Figures 8-11 NHANES I Survival Model.ipynb"),
    Path("notebooks/benchmarks/others/Benchmark Debug Mode.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/NHANES I Survival Model.ipynb"),
    Path("notebooks/benchmarks/text/Abstractive Summarization Benchmark Demo.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Performance comparison copy.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/tree_shap_paper/Figures 8-11 NHANES I Survival Model-Copy1.ipynb"),
]

allow_to_timeout = [
    Path("notebooks/tabular_examples/model_agnostic/Census income classification with scikit-learn.ipynb"),
    Path("notebooks/image_examples/image_classification/Explain MobilenetV2 using the Partition explainer (PyTorch).ipynb"),
    Path("notebooks/text_examples/sentiment_analysis/Emotion classification multiclass example.ipynb"),
    Path("notebooks/api_examples/plots/bar.ipynb"),
    Path("notebooks/text_examples/language_modelling/Language Modeling Explanation Demo.ipynb"),
    Path("notebooks/image_examples/image_classification/Explain an Intermediate Layer of VGG16 on ImageNet (PyTorch).ipynb"),
    Path("notebooks/text_examples/sentiment_analysis/Keras LSTM for IMDB Sentiment Classification.ipynb"),
    Path("notebooks/text_examples/sentiment_analysis/Positive vs. Negative Sentiment Classification.ipynb"),
    Path("notebooks/image_examples/image_classification/Multi-class ResNet50 on ImageNet (TensorFlow).ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/Fitting a Linear Simulation with XGBoost.ipynb"),
    Path("notebooks/text_examples/question_answering/Explaining a Question Answering Transformers Model.ipynb"),
    Path("notebooks/text_examples/sentiment_analysis/Using custom functions and tokenizers.ipynb"),
    Path("notebooks/text_examples/translation/Machine Translation Explanations.ipynb"),
    Path("notebooks/image_examples/image_classification/Explain an Intermediate Layer of VGG16 on ImageNet.ipynb"),
    Path("notebooks/image_examples/image_classification/Multi-class ResNet50 on ImageNet (TensorFlow)-checkpoint.ipynb"),
    Path("notebooks/api_examples/plots/text.ipynb"),
    Path("notebooks/image_examples/image_classification/Explain ResNet50 using the Partition explainer.ipynb"),
    Path("notebooks/benchmarks/tabular/Tabular Prediction Benchmark Demo.ipynb"),
    Path("notebooks/text_examples/text_generation/Open Ended GPT2 Text Generation Explanations.ipynb"),
    Path("notebooks/image_examples/image_classification/Front Page DeepExplainer MNIST Example.ipynb"),
    Path("notebooks/tabular_examples/tree_based_models/Census income classification with XGBoost.ipynb"),
    Path("notebooks/api_examples/plots/waterfall.ipynb"),
    Path("notebooks/benchmarks/text/Machine Translation Benchmark Demo.ipynb"),
    Path("notebooks/text_examples/summarization/Abstractive Summarization Explanation Demo.ipynb"),
    Path("notebooks/api_examples/plots/beeswarm.ipynb"),
    Path("notebooks/api_examples/plots/image.ipynb"),
    Path("notebooks/benchmarks/tabular/Benchmark XGBoost explanations.ipynb"),
    Path("notebooks/benchmarks/image/Image Multiclass Classification Benchmark Demo.ipynb"),
    Path("notebooks/api_examples/plots/scatter.ipynb"),
    Path("notebooks/text_examples/text_entailment/Textual Entailment Explanation Demo.ipynb"),
    Path("notebooks/image_examples/image_classification/Multi-input Gradient Explainer MNIST Example.ipynb"),
    Path("notebooks/image_examples/image_classification/PyTorch Deep Explainer MNIST example.ipynb"),
]


def main():
    notebooks_directory = Path('notebooks')
    error_notebooks = []
    ep = ExecutePreprocessor(timeout=TIMEOUT, log_level=40)

    error_notebooks = []
    timeout_notebooks = []
    notebooks_to_run = set(notebooks_directory.rglob('*.ipynb')) - set(allow_to_fail) - set(allow_to_timeout)
    for file_path in notebooks_to_run:
        with open(file_path) as f:
            nb = nbformat.read(f, as_version=4)
        start_time = time.time()
        try:
            ep.preprocess(nb, {'metadata': {'path': str(file_path.parent)}})
            print(f"Executed notebook {file_path} in {time.time() - start_time:.2f} seconds.")
        except CellExecutionError:
            error_notebooks.append(file_path)
        except TimeoutError:
            print(f"Execution of {file_path} timed out after {TIMEOUT} seconds.")
            timeout_notebooks.append(file_path)

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
