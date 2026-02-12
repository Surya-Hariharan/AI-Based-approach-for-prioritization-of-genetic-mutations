"""
Utility module for generating synthetic/dummy data for testing purposes.

This module provides functions to create synthetic genetic mutation datasets
for development, testing, and validation of the prioritization pipeline.
"""

import pandas as pd
import numpy as np
import os
from typing import Optional


def generate_synthetic_mutation_data(
    n_samples: int = 1000,
    output_path: Optional[str] = None,
    include_labels: bool = True,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic genetic mutation dataset with realistic features.
    
    Args:
        n_samples: Number of mutation samples to generate
        output_path: Path to save the generated CSV file (optional)
        include_labels: Whether to include pathogenicity labels
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame containing synthetic mutation data
        
    Example:
        >>> df = generate_synthetic_mutation_data(n_samples=500)
        >>> print(df.shape)
        (500, 6)
    """
    np.random.seed(random_seed)
    
    # Generate realistic feature distributions
    data = {
        "AF": np.random.beta(0.5, 5, n_samples),  # Allele frequency (skewed toward rare)
        "CADD_PHRED": np.random.gamma(2, 5, n_samples),  # CADD scores
        "conservation_score": np.random.beta(2, 2, n_samples),  # Conservation scores
        "variant_type": np.random.choice(["SNV", "indel"], n_samples, p=[0.85, 0.15]),
        "impact_category": np.random.choice(
            ["HIGH", "MODERATE", "LOW", "MODIFIER"], 
            n_samples, 
            p=[0.1, 0.3, 0.4, 0.2]
        ),
    }
    
    if include_labels:
        # Generate labels with some correlation to features
        # Higher CADD and conservation -> more likely pathogenic
        pathogenicity_prob = (
            0.1 + 
            0.3 * (data["CADD_PHRED"] / np.max(data["CADD_PHRED"])) +
            0.3 * data["conservation_score"] +
            0.2 * (data["impact_category"] == "HIGH")
        )
        data["pathogenicity_label"] = (
            np.random.random(n_samples) < pathogenicity_prob
        ).astype(int)
    
    df = pd.DataFrame(data)
    
    # Save if output path provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Synthetic data saved to: {output_path}")
    
    return df


def generate_test_data_with_genes(
    n_samples: int = 500,
    n_genes: int = 50,
    output_path: Optional[str] = None,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic mutation data with gene identifiers for aggregation testing.
    
    Args:
        n_samples: Number of mutation samples
        n_genes: Number of unique genes
        output_path: Path to save the generated CSV file (optional)
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with mutation data including gene IDs
    """
    df = generate_synthetic_mutation_data(
        n_samples=n_samples,
        output_path=None,
        include_labels=True,
        random_seed=random_seed
    )
    
    # Add gene identifiers (multiple variants per gene)
    np.random.seed(random_seed)
    df["gene_id"] = np.random.choice([f"GENE_{i:04d}" for i in range(n_genes)], n_samples)
    df["sample_id"] = np.random.choice([f"SAMPLE_{i:03d}" for i in range(n_samples // 10)], n_samples)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"✓ Test data with genes saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    # Generate default test datasets
    print("Generating synthetic test datasets...")
    
    # Basic feature matrix
    generate_synthetic_mutation_data(
        n_samples=1000,
        output_path="data/processed/synthetic_feature_matrix.csv"
    )
    
    # Dataset with gene IDs for aggregation testing
    generate_test_data_with_genes(
        n_samples=500,
        n_genes=50,
        output_path="data/processed/synthetic_gene_data.csv"
    )
    
    print("\n✓ All synthetic datasets generated successfully!")
