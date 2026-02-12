import json
import os
from typing import Dict, Any

class EvaluationReport:
    def __init__(self, output_dir: str = "reports/results"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.report_data = {}

    def add_metrics(self, metrics: Dict[str, Any]):
        self.report_data['metrics'] = metrics

    def add_plot_paths(self, plot_paths: Dict[str, str]):
        self.report_data['plots'] = plot_paths
        
    def add_config(self, config_dict: Dict[str, Any]):
        self.report_data['config'] = config_dict

    def save(self, filename: str = "evaluation_report.json"):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.report_data, f, indent=4)
        print(f"Report saved to {path}")
