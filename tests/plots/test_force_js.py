"""Tests for force plot using JavaScript/React rendering with Selenium."""

import os
import platform
import tempfile
import time

import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

import shap
from shap.plots._force import save_html

# Skip if selenium is not installed
pytest.importorskip("selenium")
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


@pytest.fixture(scope="module")
def driver():
    """Setup headless Chrome/Chromium for testing."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")

    # Try with both 'chrome' and 'chromium' executable names
    try:
        driver = webdriver.Chrome(options=options)
    except Exception:
        try:
            driver = webdriver.Chrome(service=Service("/usr/bin/chromium"), options=options)
        except Exception:
            pytest.skip("Chrome/Chromium not available for Selenium tests")

    driver.set_window_size(1000, 600)
    yield driver
    driver.quit()


def get_sample_force_plot():
    """Create a sample force plot for testing."""
    # Create a simple model
    X, y = shap.datasets.adult(n_points=100)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X, y)

    # Create the explainer
    ex = shap.TreeExplainer(model)
    shap_values = ex(X)

    # Get the first instance's explanation
    return shap.plots.force(ex.expected_value[0], shap_values.values[:, 0])


def get_sample_force_array_plot():
    """Create a sample force array plot with multiple instances for testing."""
    # Create a simple model
    X, y = shap.datasets.adult(n_points=100)
    model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=0)
    model.fit(X, y)

    # Create the explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    # Get multiple explanations
    return shap.plots.force(explainer.expected_value[0], shap_values.values[:5, 0])


def capture_plot_screenshot(driver, plot, filename=None, wait_time=2):
    """
    Render a force plot and capture a screenshot using Selenium.

    Parameters
    ----------
    driver : selenium.webdriver
        The selenium webdriver instance
    plot : AdditiveForceVisualizer or AdditiveForceArrayVisualizer
        The SHAP force plot to render
    filename : str, optional
        Path where to save the HTML file, if None a temporary file is used
    wait_time : int, optional
        Time to wait for JavaScript execution in seconds

    Returns
    -------
    PIL.Image
        The screenshot image
    """
    import io

    # Create temp HTML file if no filename provided
    if filename is None:
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as tmp:
            filename = tmp.name

    # Save the force plot as HTML
    save_html(filename, plot)

    # Open the HTML in the browser
    driver.get(f"file://{filename}")

    # Wait for the plot to render
    time.sleep(wait_time)

    # Take a screenshot
    screenshot = driver.get_screenshot_as_png()

    # Clean up the temp file
    if not filename.startswith(os.path.join(os.path.dirname(__file__), "baseline")):
        os.unlink(filename)

    # Convert to PIL image
    return Image.open(io.BytesIO(screenshot))


@pytest.mark.skipif(platform.system() == "Windows", reason="Selenium force plot tests have different sizes on windows.")
def test_force_js_visual(driver):
    """Test that force plot renders correctly."""
    # Create directory for baseline images if it doesn't exist
    baseline_dir = os.path.join(os.path.dirname(__file__), "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Get sample force plot
    plot = get_sample_force_plot()

    # Define baseline path
    baseline_path = os.path.join(baseline_dir, "test_force_js.png")

    # Capture screenshot
    screenshot = capture_plot_screenshot(driver, plot)

    # If baseline doesn't exist, save current screenshot as baseline
    if not os.path.exists(baseline_path):
        screenshot.save(baseline_path)
        pytest.skip(f"Baseline image created at {baseline_path}")

    # Compare with baseline
    baseline = Image.open(baseline_path)

    # Ensure same dimensions
    assert screenshot.size == baseline.size, "Screenshot dimensions don't match baseline"

    # Convert to numpy arrays for comparison
    screenshot_array = np.array(screenshot)
    baseline_array = np.array(baseline)

    # Calculate difference (allowing for some variation)
    diff = np.mean(np.abs(screenshot_array.astype(float) - baseline_array.astype(float)))
    assert diff < 10.0, f"Images differ by {diff} average pixel value"


@pytest.mark.skipif(platform.system() == "Windows", reason="Selenium force plot tests have different sizes on windows.")
def test_force_array_js_visual(driver):
    """Test that force array plot renders correctly."""
    # Create directory for baseline images if it doesn't exist
    baseline_dir = os.path.join(os.path.dirname(__file__), "baseline")
    os.makedirs(baseline_dir, exist_ok=True)

    # Get sample force array plot
    plot = get_sample_force_array_plot()

    # Define baseline path
    baseline_path = os.path.join(baseline_dir, "test_force_array_js.png")

    # Capture screenshot
    screenshot = capture_plot_screenshot(driver, plot, wait_time=3)  # Array plot might need more time

    # If baseline doesn't exist, save current screenshot as baseline
    if not os.path.exists(baseline_path):
        screenshot.save(baseline_path)
        pytest.skip(f"Baseline image created at {baseline_path}")

    # Compare with baseline
    baseline = Image.open(baseline_path)

    # Ensure same dimensions
    assert screenshot.size == baseline.size, "Screenshot dimensions don't match baseline"

    # Convert to numpy arrays for comparison
    screenshot_array = np.array(screenshot)
    baseline_array = np.array(baseline)

    # Calculate difference (allowing for some variation)
    diff = np.mean(np.abs(screenshot_array.astype(float) - baseline_array.astype(float)))
    assert diff < 15.0, f"Images differ by {diff} average pixel value"
