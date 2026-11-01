import nbformat as nbf
from nbclient import NotebookClient

nb = nbf.read("Telco_Churn_Prediction.ipynb", as_version=4)

client = NotebookClient(nb, timeout=600, kernel_name="python3")
try:
    client.execute()
    print("NOTEBOOK EXECUTED SUCCESSFULLY - no errors")
except Exception as e:
    print(f"NOTEBOOK EXECUTION FAILED: {e}")
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "code":
            for output in cell.get("outputs", []):
                if output.get("output_type") == "error":
                    print(f"  Cell {i} ERROR: {output.get('ename', '')}: {output.get('evalue', '')}")
    raise

# Save with outputs (R4)
nbf.write(nb, "Telco_Churn_Prediction.ipynb")
print("Notebook saved with outputs.")

# Verify all code cells have outputs
code_cells = [c for c in nb.cells if c.cell_type == "code"]
cells_with_outputs = sum(1 for c in code_cells if c.get("outputs"))
cells_with_exec = sum(1 for c in code_cells if c.get("execution_count") is not None)
print(f"Code cells: {len(code_cells)}")
print(f"Code cells with outputs: {cells_with_outputs}")
print(f"Code cells with execution_count: {cells_with_exec}")
