import sys
import os
import runpy

# Add root directory to python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

print("Running Synthetic Data Demo...")
runpy.run_path("src/08_visualiser.py", run_name="__main__")
print("All demo plots generated in output/plots/")
