"""
Data validation utilities for ensuring data quality and consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates dataset structure, completeness, and quality.
    """
    
    def __init__(self, required_columns: Optional[List[str]] = None, 
                 target_col: Optional[str] = None):
        """
        Initialize validator.
        
        Args:
            required_columns: List of columns that must be present
            target_col: Name of the target column
        """
        self.required_columns = required_columns or []
        self.target_col = target_col
        self.validation_report = {}
        
    def validate(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict[str, Any]:
        """
        Run comprehensive validation checks on dataset.
        
        Args:
            df: DataFrame to validate
            dataset_name: Name for logging purposes
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating {dataset_name}...")
        
        self.validation_report = {
            'dataset_name': dataset_name,
            'shape': df.shape,
            'columns': list(df.columns),
            'checks': {}
        }
        
        # Check 1: Required columns exist
        self._check_required_columns(df)
        
        # Check 2: Target column exists
        self._check_target_column(df)
        
        # Check 3: No duplicate column names
        self._check_duplicate_columns(df)
        
        # Check 4: Missing value analysis
        self._check_missing_values(df)
        
        # Check 5: Data types
        self._check_data_types(df)
        
        # Check 6: Dataset not empty
        self._check_not_empty(df)
        
        # Log summary
        self._log_summary()
        
        return self.validation_report
    
    def _check_required_columns(self, df: pd.DataFrame):
        """Check all required columns are present."""
        missing_cols = set(self.required_columns) - set(df.columns)
        
        if missing_cols:
            self.validation_report['checks']['required_columns'] = {
                'status': 'FAILED',
                'message': f"Missing required columns: {missing_cols}"
            }
            logger.error(f"❌ Missing required columns: {missing_cols}")
        else:
            self.validation_report['checks']['required_columns'] = {
                'status': 'PASSED',
                'message': 'All required columns present'
            }
            logger.info("✓ All required columns present")
    
    def _check_target_column(self, df: pd.DataFrame):
        """Check target column exists."""
        if self.target_col:
            if self.target_col not in df.columns:
                self.validation_report['checks']['target_column'] = {
                    'status': 'FAILED',
                    'message': f"Target column '{self.target_col}' not found"
                }
                logger.error(f"❌ Target column '{self.target_col}' not found")
            else:
                self.validation_report['checks']['target_column'] = {
                    'status': 'PASSED',
                    'message': f"Target column '{self.target_col}' found",
                    'unique_values': df[self.target_col].nunique(),
                    'value_counts': df[self.target_col].value_counts().to_dict()
                }
                logger.info(f"✓ Target column '{self.target_col}' present with {df[self.target_col].nunique()} unique values")
    
    def _check_duplicate_columns(self, df: pd.DataFrame):
        """Check for duplicate column names."""
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        
        if duplicate_cols:
            self.validation_report['checks']['duplicate_columns'] = {
                'status': 'FAILED',
                'message': f"Duplicate column names: {duplicate_cols}"
            }
            logger.error(f"❌ Duplicate column names: {duplicate_cols}")
        else:
            self.validation_report['checks']['duplicate_columns'] = {
                'status': 'PASSED',
                'message': 'No duplicate column names'
            }
            logger.info("✓ No duplicate column names")
    
    def _check_missing_values(self, df: pd.DataFrame):
        """Analyze missing values."""
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df) * 100).round(2)
        
        cols_with_missing = missing_stats[missing_stats > 0]
        
        if len(cols_with_missing) > 0:
            missing_info = {
                col: {
                    'count': int(missing_stats[col]),
                    'percentage': float(missing_pct[col])
                }
                for col in cols_with_missing.index
            }
            
            self.validation_report['checks']['missing_values'] = {
                'status': 'WARNING',
                'message': f"{len(cols_with_missing)} columns have missing values",
                'details': missing_info
            }
            logger.warning(f"⚠ {len(cols_with_missing)} columns have missing values")
            for col, info in missing_info.items():
                logger.warning(f"  - {col}: {info['count']} ({info['percentage']}%)")
        else:
            self.validation_report['checks']['missing_values'] = {
                'status': 'PASSED',
                'message': 'No missing values'
            }
            logger.info("✓ No missing values")
    
    def _check_data_types(self, df: pd.DataFrame):
        """Check and report data types."""
        dtypes_summary = df.dtypes.value_counts().to_dict()
        dtypes_summary = {str(k): int(v) for k, v in dtypes_summary.items()}
        
        self.validation_report['checks']['data_types'] = {
            'status': 'INFO',
            'summary': dtypes_summary,
            'details': df.dtypes.astype(str).to_dict()
        }
        logger.info(f"ℹ Data types: {dtypes_summary}")
    
    def _check_not_empty(self, df: pd.DataFrame):
        """Check dataset is not empty."""
        if df.empty:
            self.validation_report['checks']['not_empty'] = {
                'status': 'FAILED',
                'message': 'Dataset is empty'
            }
            logger.error("❌ Dataset is empty")
        else:
            self.validation_report['checks']['not_empty'] = {
                'status': 'PASSED',
                'message': f'Dataset contains {len(df)} rows'
            }
            logger.info(f"✓ Dataset contains {len(df)} rows")
    
    def _log_summary(self):
        """Log validation summary."""
        logger.info(f"\n{'='*60}")
        logger.info(f"Validation Summary for {self.validation_report['dataset_name']}")
        logger.info(f"{'='*60}")
        logger.info(f"Shape: {self.validation_report['shape']}")
        logger.info(f"Columns: {len(self.validation_report['columns'])}")
        
        passed = sum(1 for check in self.validation_report['checks'].values() 
                     if check['status'] == 'PASSED')
        failed = sum(1 for check in self.validation_report['checks'].values() 
                     if check['status'] == 'FAILED')
        warnings = sum(1 for check in self.validation_report['checks'].values() 
                       if check['status'] == 'WARNING')
        
        logger.info(f"Passed: {passed} | Failed: {failed} | Warnings: {warnings}")
        logger.info(f"{'='*60}\n")
    
    def raise_on_failure(self):
        """Raise exception if any validation checks failed."""
        failed_checks = [
            check_name for check_name, check_result in self.validation_report['checks'].items()
            if check_result['status'] == 'FAILED'
        ]
        
        if failed_checks:
            raise ValueError(
                f"Validation failed for checks: {failed_checks}\n"
                f"See validation report for details."
            )


def validate_and_save(df: pd.DataFrame, 
                     filepath: Path, 
                     validator: DataValidator,
                     dataset_name: str = "Dataset") -> pd.DataFrame:
    """
    Validate dataset and save if validation passes.
    
    Args:
        df: DataFrame to validate and save
        filepath: Path to save the validated dataset
        validator: DataValidator instance
        dataset_name: Name for logging
        
    Returns:
        Validated DataFrame
        
    Raises:
        ValueError: If validation fails
    """
    # Ensure filepath is Path object
    filepath = Path(filepath)
    
    # SAFETY: Prevent writing to raw/ directory
    if 'raw' in filepath.parts:
        raise PermissionError(
            f"Cannot write to raw/ directory: {filepath}\n"
            "Raw data must remain untouched. Use interim/ or processed/ instead."
        )
    
    # Validate
    validator.validate(df, dataset_name=dataset_name)
    validator.raise_on_failure()
    
    # Create parent directory if needed
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Save
    df.to_csv(filepath, index=False)
    logger.info(f"✓ Saved {dataset_name} to {filepath}")
    
    return df
