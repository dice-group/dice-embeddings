"""
Comparison example: Triple-Centric vs Entity-Centric approaches

This script demonstrates both approaches on a small example KG.
"""

import numpy as np
import pandas as pd
from tabpfn_over_graph import KGToTabularConverter, EntityCentricConverter

# Example Knowledge Graph
example_triples = [
    ('CaglarDemir', 'LivesIn', 'Germany'),
    ('CaglarDemir', 'isA', 'ComputerScientist'),
    ('CaglarDemir', 'isA', 'Person'),
    ('Germany', 'isA', 'Country'),
    ('Germany', 'locatedIn', 'Europe'),
    ('Alice', 'LivesIn', 'USA'),
    ('Alice', 'isA', 'Person'),
    ('USA', 'isA', 'Country'),
]

print("="*80)
print("Knowledge Graph to Tabular Conversion: Comparison Example")
print("="*80)

print("\nInput Knowledge Graph:")
print("-" * 40)
for h, r, t in example_triples:
    print(f"  {h:20s} {r:15s} {t}")

# ============================================================================
# APPROACH 1: Triple-Centric (Original)
# ============================================================================
print("\n" + "="*80)
print("APPROACH 1: Triple-Centric Representation")
print("="*80)
print("\nEach row represents a TRIPLE with computed graph features")

converter1 = KGToTabularConverter()

# Build vocabulary and graph structure
converter1.build_vocabulary(example_triples)
converter1.build_graph_structure(example_triples)

# Convert to tabular format
X_triple, y_triple = converter1.triples_to_tabular(example_triples)

print(f"\nShape: {X_triple.shape}")
print(f"Columns: h_idx, r_idx, t_idx, h_out_deg, h_in_deg, ...")

# Show first few rows
print("\nSample rows (first 5):")
feature_names = ['h_idx', 'r_idx', 't_idx', 'h_out_deg', 'h_in_deg', 
                 'h_num_out_rel', 'h_num_in_rel', 'h_avg_out_nbr', 'h_total_deg',
                 't_out_deg', 't_in_deg', 't_num_out_rel', 't_num_in_rel', 
                 't_avg_in_nbr', 't_total_deg', 'r_freq']
df_triple = pd.DataFrame(X_triple[:5], columns=feature_names)
print(df_triple.to_string(index=False))

print("\nInterpretation:")
print("  - Each row is a triple (h,r,t)")
print("  - Features describe graph topology")
print("  - Fixed feature size regardless of number of relations")
print("  - Good for: Link prediction, graph structure analysis")

# ============================================================================
# APPROACH 2: Entity-Centric (New)
# ============================================================================
print("\n" + "="*80)
print("APPROACH 2: Entity-Centric Representation")
print("="*80)
print("\nEach row represents an ENTITY with all its relations as columns")

converter2 = EntityCentricConverter()

# Convert to entity-centric format
df_entity = converter2.triples_to_entity_centric_tabular(example_triples)

print(f"\nShape: {df_entity.shape}")
print(f"Columns: {list(df_entity.columns)}")

print("\nFull table:")
print(df_entity.to_string(index=False))

print("\nInterpretation:")
print("  - Each row represents an entity")
print("  - Columns are relation types")
print("  - Entities can appear multiple times (for multi-valued relations)")
print("  - 'NotApplicable' means the relation doesn't apply to that entity")
print("  - Good for: Entity classification, property prediction")

# ============================================================================
# Key Differences
# ============================================================================
print("\n" + "="*80)
print("KEY DIFFERENCES")
print("="*80)

print(f"\nNumber of rows:")
print(f"  Triple-Centric: {len(X_triple)} (one per triple)")
print(f"  Entity-Centric: {len(df_entity)} (entities can have multiple rows)")

print(f"\nNumber of features/columns:")
print(f"  Triple-Centric: {X_triple.shape[1]} (fixed graph features)")
print(f"  Entity-Centric: {df_entity.shape[1]} (one per relation type + Label)")

print("\nData representation:")
print("  Triple-Centric: Numerical features describing graph structure")
print("  Entity-Centric: Categorical values showing relation-object pairs")

print("\nBest use cases:")
print("  Triple-Centric:")
print("    - Link prediction (will this triple exist?)")
print("    - Leveraging graph topology")
print("    - When graph structure matters")
print("\n  Entity-Centric:")
print("    - Entity classification (what type is this entity?)")
print("    - Property prediction (what values does this entity have?)")
print("    - When relations matter more than topology")

# ============================================================================
# Example Queries
# ============================================================================
print("\n" + "="*80)
print("EXAMPLE QUERIES")
print("="*80)

print("\nTriple-Centric Query: 'What are the features of (CaglarDemir, LivesIn, Germany)?'")
caglar_germany_idx = 0  # First row in our example
print(f"  h_idx={X_triple[caglar_germany_idx, 0]:.0f}, "
      f"r_idx={X_triple[caglar_germany_idx, 1]:.0f}, "
      f"t_idx={X_triple[caglar_germany_idx, 2]:.0f}")
print(f"  Head out-degree: {X_triple[caglar_germany_idx, 3]:.0f}")
print(f"  Head in-degree: {X_triple[caglar_germany_idx, 4]:.0f}")

print("\nEntity-Centric Query: 'What are all the properties of CaglarDemir?'")
caglar_rows = df_entity[df_entity['Entity'] == 'CaglarDemir']
print(caglar_rows.to_string(index=False))

print("\nEntity-Centric Query: 'Which entities have LivesIn relation?'")
entities_with_livesin = df_entity[df_entity['LivesIn'] != 'NotApplicable'][['Entity', 'LivesIn']]
print(entities_with_livesin.to_string(index=False))

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("""
Both approaches convert the same knowledge graph to tabular format, but with
different perspectives:

1. Triple-Centric: Treats each triple as a data point with computed features
   - More aligned with traditional KG embedding approaches
   - Better for link prediction tasks
   
2. Entity-Centric: Treats each entity as a data point with relation-value pairs
   - More aligned with relational database thinking
   - Better for entity-focused tasks
   
Choose based on your task:
- Link Prediction? Use Triple-Centric
- Entity Classification? Use Entity-Centric
- Unsure? Try both and compare!
""")
