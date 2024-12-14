import pandas as pd
import numpy as np
from collections import Counter

# Load data
print("Loading data...")
train_terms = pd.read_csv("CAFA 5 Protein Function Prediction/Train/train_terms.tsv", sep="\t")

# Analyze GO term frequencies
term_counts = Counter(train_terms['term'])
print("\nGO term statistics:")
print(f"Total unique GO terms: {len(term_counts)}")
print(f"Most common terms appear: {sorted(term_counts.values(), reverse=True)[:5]} times")
print(f"Least common terms appear: {sorted(term_counts.values())[:5]} times")

# Analyze protein annotation counts
protein_counts = Counter(train_terms['EntryID'])
print("\nProtein annotation statistics:")
print(f"Total unique proteins: {len(protein_counts)}")
print(f"Most annotated proteins have: {sorted(protein_counts.values(), reverse=True)[:5]} annotations")
print(f"Least annotated proteins have: {sorted(protein_counts.values())[:5]} annotations")

# Let's create a more balanced dataset by:
# 1. Keeping only GO terms that appear at least X times
# 2. Keeping only proteins that have at least Y annotations

min_term_occurrences = 100 
min_protein_annotations = 10  

# Filter frequent terms
frequent_terms = {term for term, count in term_counts.items() if count >= min_term_occurrences}
print(f"\nTerms occurring at least {min_term_occurrences} times: {len(frequent_terms)}")

# Filter well-annotated proteins
well_annotated_proteins = {prot for prot, count in protein_counts.items() if count >= min_protein_annotations}
print(f"Proteins with at least {min_protein_annotations} annotations: {len(well_annotated_proteins)}")

# Create balanced dataset
balanced_terms = train_terms[
    (train_terms['term'].isin(frequent_terms)) & 
    (train_terms['EntryID'].isin(well_annotated_proteins))
]

print("\nBalanced dataset statistics:")
print(f"Original dataset size: {len(train_terms)}")
print(f"Balanced dataset size: {len(balanced_terms)}")

# Save balanced dataset
balanced_terms.to_csv("balanced_train_terms.tsv", sep="\t", index=False)

# Calculate final class balance
print("\nFinal class balance analysis...")
unique_terms_balanced = balanced_terms['term'].unique()
unique_proteins_balanced = balanced_terms['EntryID'].unique()

labels_balanced = np.zeros((len(unique_proteins_balanced), len(unique_terms_balanced)))
term_to_idx = {term: idx for idx, term in enumerate(unique_terms_balanced)}
protein_to_idx = {pid: idx for idx, pid in enumerate(unique_proteins_balanced)}

for _, row in balanced_terms.iterrows():
    if row['EntryID'] in protein_to_idx:
        protein_idx = protein_to_idx[row['EntryID']]
        term_idx = term_to_idx[row['term']]
        labels_balanced[protein_idx, term_idx] = 1

print(f"\nBalanced label matrix shape: {labels_balanced.shape}")
print(f"Ones: {np.sum(labels_balanced == 1)}")
print(f"Zeros: {np.sum(labels_balanced == 0)}")
print(f"Ratio of ones: {np.sum(labels_balanced == 1) / labels_balanced.size:.4f}")