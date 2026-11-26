import pytest


class WarningCollector:
    def pytest_warning_recorded(self, warning_message, when, nodeid, location):
        if issubclass(warning_message.category, (DeprecationWarning, FutureWarning)):
            with open("deprecation_warnings.log", "a") as f:
                f.write(
                    f"{warning_message.category.__name__}: {warning_message.message} at {warning_message.filename}:{warning_message.lineno}\n"
                )


if __name__ == "__main__":
    with open("deprecation_warnings.log", "w") as f:
        f.write("")
    pytest.main(["tests/"], plugins=[WarningCollector()])
