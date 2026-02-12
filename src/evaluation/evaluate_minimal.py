import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

try:
    from src.evaluation.eval_metrics import calculate_top_k_recall
    print("SUCCESS: Imported calculate_top_k_recall")
    import src.evaluation.eval_metrics as m
    print(f"File: {m.__file__}")

    from src.evaluation.plotting import Plotter
    print("SUCCESS: Imported Plotter")

except ImportError as e:
    print(f"FAILURE: {e}")
except Exception as e:
    print(f"EXCEPTION: {e}")
