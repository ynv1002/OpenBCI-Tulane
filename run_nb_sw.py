import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import glob

try:
    notebook_path = glob.glob('/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/*.ipynb')[0]
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': '/Users/yanivnaggar/Desktop/Spring 2026/IS/BCI-project/'}})

    print("--- 1s Sliding Window Output ---")
    
    # Print the last cell output which contains our new sliding window code
    last_cell = nb.cells[-1]
    if last_cell.cell_type == 'code':
        for output in last_cell.outputs:
            if output.output_type == 'stream':
                print(output.text)
            elif output.output_type in ('display_data', 'execute_result'):
                if 'text/plain' in output.data:
                    print(output.data['text/plain'])
            elif output.output_type == 'error':
                print(f"Error: {output.ename} - {output.evalue}")
                for trace in output.traceback:
                    import re
                    print(re.sub(r'\x1b\[.*?m', '', trace))
                    
except Exception as e:
    import traceback
    traceback.print_exc()
