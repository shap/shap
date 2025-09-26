"""The plotting baseline folder is generated using the pytest-mpl plugin.

If you have made changes to the plots, the baseline folder will need rebuilding
before the tests can run successfully. Run the following in the root directory:

`pytest tests/plots --mpl-generate-path=tests/plots/baseline`

then check that the plot images in the baseline folder are the expected output.

Then tests can be ran using `pytest .`, and the plots generated from the plot
tests will then be compared to the baseline.
"""
