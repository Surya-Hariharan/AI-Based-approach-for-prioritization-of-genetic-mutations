import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    import src.evaluation.eval_metrics as m
    print(f"Imported metrics from: {m.__file__}")
    print(f"calculate_top_k_recall in metrics: {'calculate_top_k_recall' in dir(m)}")
except Exception as e:
    print(f"Error importing metrics: {e}")
    import traceback
    traceback.print_exc()

try:
    from src.evaluation.eval_metrics import calculate_top_k_recall
    print("Successfully imported calculate_top_k_recall")
except ImportError as e:
    print(f"ImportError: {e}")
