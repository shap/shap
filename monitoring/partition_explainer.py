from asv_runner.benchmarks.mark import skip_benchmark_if
from xgboost import XGBClassifier

from shap import Explanation
from shap.datasets import adult, imagenet50
from shap.explainers import PartitionExplainer
from shap.maskers import Image

try:
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
except ImportError:
    TENSORFLOW_AVAILABLE = False
else:
    TENSORFLOW_AVAILABLE = True


class PartitionSuite:
    # Adapted from tests/explainers/test_partition.py
    # TODO: should we add translation tests here too?
    # This would introduce a dependency on torch and transformers.
    max_samples = 100

    def setup(self):
        self.model = XGBClassifier(tree_method="exact", base_score=0.5)

        # get a dataset on income prediction
        self.X, self.y = adult()
        if self.max_samples is not None:
            self.X = self.X.iloc[: self.max_samples]
            self.y = self.y[: self.max_samples]
        self.X = self.X.values

        # fit the model on the data
        self.model.fit(self.X, self.y)

    def time_single_output(self):
        ex = PartitionExplainer(self.model.predict, self.X)
        _ = ex(self.X)

    def time_multi_output(self):
        ex = PartitionExplainer(self.model.predict_proba, self.X)
        _ = ex(self.X)


class ImagePartitionSuite:
    # Adapted from "Image Data Explanation Benchmarking: Image Multiclass Classification" notebook

    def setup(self):
        if TENSORFLOW_AVAILABLE:
            self.model = ResNet50(weights="imagenet")
        self.X, self.y = imagenet50()

        # Dry run to avoid numba jit compilation time in the benchmark
        _ = Image("inpaint_telea", self.X[0].shape)

    def predict(self, x):
        tmp = x.copy()
        if len(tmp.shape) == 2:
            tmp = tmp.reshape(tmp.shape[0], *self.X[0].shape)
        tmp = preprocess_input(tmp)
        return self.model(tmp)

    def time_masker_init(self):
        _ = Image("inpaint_telea", self.X[0].shape)

    @skip_benchmark_if(not TENSORFLOW_AVAILABLE)
    def time_multi_output(self):
        # Skipping this test for now because it is very slow and adds a dependency on tensorflow
        masker = Image("inpaint_telea", self.X[0].shape)
        explainer = PartitionExplainer(self.predict, masker)
        _ = explainer(self.X[1:3], max_evals=500, batch_size=50, outputs=Explanation.argsort.flip[:4])
