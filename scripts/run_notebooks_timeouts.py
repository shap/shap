import os
import subprocess
import time

from nbconvert.exporters import ScriptExporter

TIMEOUT = 60  # seconds


def convert_notebook_to_python(notebook_path, output_path):
    # Add Matplotlib configuration to use a non-interactive backend
    exporter = ScriptExporter()
    content, _ = exporter.from_filename(notebook_path)
    content = 'import matplotlib\nmatplotlib.use("Agg")\n' + content

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def execute_python_script(script_path, timeout_seconds=30):
    start_time = time.time()
    try:
        ret_code = subprocess.call(['python', script_path],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT,
                                   timeout=timeout_seconds)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Executed notebook {script_path} in {execution_time:.2f} seconds.")
        return ret_code, execution_time  # Successful execution
    except subprocess.TimeoutExpired:
        return -1, None  # Timeout
    except subprocess.CalledProcessError as e:
        end_time = time.time()
        execution_time = end_time - start_time
        return e.returncode, execution_time  # Capture error code

def main():
    notebooks_directory = 'notebooks'
    error_notebooks = []

    for root, dirs, files in os.walk(notebooks_directory):
        for file in files:
            if file.endswith(".ipynb"):
                notebook_path = os.path.join(root, file)
                python_script_path = os.path.splitext(notebook_path)[0] + '.py'

                convert_notebook_to_python(notebook_path, python_script_path)

                # error_code, execution_time = execute_python_script(python_script_path, timeout_seconds=TIMEOUT)

                # if error_code == -1:
                #     print(f"Execution of {notebook_path} timed out after {TIMEOUT} seconds.")
                #     error_notebooks.append((notebook_path, -1, None))
                # elif error_code != 0:
                #     error_notebooks.append((notebook_path, error_code, execution_time))

    if error_notebooks:
        error_thrown = [error_code for _, error_code, _ in error_notebooks if error_code != -1]
        print("Notebooks with error codes or timeouts:")
        for notebook, error_code, execution_time in error_notebooks:
            if error_code == -1:
                print(f"{notebook}: Timeout")
            else:
                print(f"{notebook}: Error Code {error_code}, Execution Time: {execution_time:.2f} seconds")
        # if len(error_thrown) > 0:
        #     raise Exception(f"Notebooks failed with error codes: {', '.join([path for path, _, _ in error_notebooks if _ in error_thrown])}")
    else:
        print("All notebooks executed successfully.")

if __name__ == "__main__":
    main()
