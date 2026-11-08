"""Execute run.ipynb in place (Run All). Run: .venv/bin/python -m gru_classifier.run_notebook"""
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).with_name("run.ipynb")

if __name__ == "__main__":
    nb = nbformat.read(NB, as_version=4)
    n = sum(c.cell_type == "code" for c in nb.cells)
    t0, done = time.time(), [0]

    def on_cell_executed(cell, cell_index, execute_reply):
        if cell.cell_type == "code":
            done[0] += 1
            print(f"[{done[0]}/{n}] {time.time() - t0:5.0f}s  {cell.source.splitlines()[0][:70]}", flush=True)

    NotebookClient(nb, timeout=None, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}},
                   on_cell_executed=on_cell_executed).execute()
    nbformat.write(nb, NB)
    print(f"done in {time.time() - t0:.0f}s -> {NB}")
