import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
try:
    import src.evaluation.eval_metrics as m
    print(f"File: {m.__file__}")
    from sklearn.metrics import roc_curve
    print("SUCCESS: Imported roc_curve")
except ImportError as e:
    print(f"FAILURE: {e}")
