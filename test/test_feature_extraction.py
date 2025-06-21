import numpy as np
import sys
from os.path import realpath, join, split

# Ensure gqsat module can be found
# This assumes the script is run from the root of the CGSAT project
sys.path.append(realpath(join(split(realpath(__file__))[0], '.')))
from gqsat.cnf_features import CNFFeatureExtractor

# --- Constants (mirroring MiniSATEnv.py for consistency) ---
# Feature column indices
NODE_TYPE_COL = 0
NODE_ID_COL = 1
HANDCRAFTED_FEATURES_START_COL = 2

# Handcrafted feature dimensions
NUM_HANDCRAFTED_VAR_FEATURES = 5
NUM_HANDCRAFTED_CLAUSE_FEATURES = 15

# Node type identifiers
NODE_TYPE_VAR = 1
NODE_TYPE_CLAUSE = 2

# Edge feature dimension
EDGE_IN_SIZE = 2

def generate_graph_features_for_cnf(
    total_vars_in_cnf, 
    clauses_in_cnf, 
    var_embedding_dim=16, 
    clause_embedding_dim=16
):
    """
    Generates graph feature matrices for a given CNF problem.
    Assumes all variables are initially unassigned.

    Args:
        total_vars_in_cnf (int): Total number of variables in the CNF problem (e.g., 3 for vars x1, x2, x3).
        clauses_in_cnf (list of list of int): List of clauses.
            Each clause is a list of literals.
            Literals are 1-indexed; sign indicates polarity (e.g., [[1, -2], [2, 3]]).
        var_embedding_dim (int): Dimension for variable embeddings.
        clause_embedding_dim (int): Dimension for clause embeddings.

    Returns:
        tuple: (vertex_data, edge_data, connectivity)
    """

    # Calculate total feature dimension for nodes
    var_features_total_dim = (
        1  # Type
        + 1  # ID
        + NUM_HANDCRAFTED_VAR_FEATURES
        + var_embedding_dim
    )
    clause_features_total_dim = (
        1  # Type
        + 1  # ID
        + NUM_HANDCRAFTED_CLAUSE_FEATURES
        + clause_embedding_dim
    )
    vertex_in_size = max(var_features_total_dim, clause_features_total_dim)

    # --- Simulate MiniSATEnv state for an initial, unsolved problem ---
    # All variables are initially unassigned
    num_unassigned_vars = total_vars_in_cnf
    # Original 0-indexed variable IDs for unassigned variables
    unassigned_var_original_ids = list(range(total_vars_in_cnf))

    # Mapping from original 0-indexed var ID to compact 0-indexed ID in the graph
    # For this initial state, it's an identity mapping for all variables
    vars_remapping_compact_to_original = {i: i for i in unassigned_var_original_ids}
    vars_remapping_original_to_compact = {original_id: compact_id for compact_id, original_id in vars_remapping_compact_to_original.items()}

    num_clauses = len(clauses_in_cnf)

    # --- Initialize graph data structures ---
    vertex_data = np.zeros(
        (num_unassigned_vars + num_clauses, vertex_in_size), dtype=np.float32
    )

    # Calculate number of edges (each literal in a clause creates two directed edges)
    num_literals_total = sum(len(cl) for cl in clauses_in_cnf)
    edge_data = np.zeros((num_literals_total * 2, EDGE_IN_SIZE), dtype=np.float32)
    connectivity = np.zeros((2, num_literals_total * 2), dtype=np.int_)

    # --- Populate Edges and Connectivity ---
    edge_counter = 0
    for clause_idx, clause_literals in enumerate(clauses_in_cnf):
        compact_clause_node_idx = num_unassigned_vars + clause_idx
        for lit in clause_literals:
            original_var_id_0_indexed = abs(lit) - 1
            compact_var_node_idx = vars_remapping_original_to_compact[original_var_id_0_indexed]

            # Edge feature: [1,0] for positive literal, [0,1] for negative literal
            # This matches the logic: edge_data[ec:ec+2, int(l > 0)] = 1
            # If l > 0 (positive), int(l > 0) is 1. Edge feature is [?, 1]. So [0,1]
            # If l < 0 (negative), int(l > 0) is 0. Edge feature is [1, ?]. So [1,0]
            if lit > 0: # Positive literal
                edge_data[edge_counter : edge_counter + 2, 1] = 1
            else: # Negative literal
                edge_data[edge_counter : edge_counter + 2, 0] = 1
            
            # From variable to clause
            connectivity[0, edge_counter] = compact_var_node_idx
            connectivity[1, edge_counter] = compact_clause_node_idx
            # From clause to variable
            connectivity[0, edge_counter + 1] = compact_clause_node_idx
            connectivity[1, edge_counter + 1] = compact_var_node_idx
            edge_counter += 2

    # --- Populate Vertex Data ---
    # 1. Node Types
    vertex_data[:num_unassigned_vars, NODE_TYPE_COL] = NODE_TYPE_VAR
    vertex_data[num_unassigned_vars:, NODE_TYPE_COL] = NODE_TYPE_CLAUSE

    # 2. Node Original IDs
    for i in range(num_unassigned_vars):
        vertex_data[i, NODE_ID_COL] = unassigned_var_original_ids[i] # Original 0-indexed var ID
    for i in range(num_clauses):
        vertex_data[num_unassigned_vars + i, NODE_ID_COL] = i # 0-indexed clause ID

    # 3. Handcrafted Features (using CNFFeatureExtractor)
    # CNFFeatureExtractor expects 1-indexed variables in clauses
    # clauses_in_cnf is already in this format.
    feature_extractor = CNFFeatureExtractor(clauses_in_cnf, total_vars_in_cnf)

    var_handcrafted_features = feature_extractor.extract_var_features()
    for compact_idx, original_idx in vars_remapping_compact_to_original.items():
        start_col = HANDCRAFTED_FEATURES_START_COL
        end_col = start_col + NUM_HANDCRAFTED_VAR_FEATURES
        vertex_data[compact_idx, start_col:end_col] = var_handcrafted_features[original_idx]

    clause_handcrafted_features = feature_extractor.extract_clause_features()
    start_col = HANDCRAFTED_FEATURES_START_COL
    end_col = start_col + NUM_HANDCRAFTED_CLAUSE_FEATURES
    vertex_data[num_unassigned_vars:, start_col:end_col] = clause_handcrafted_features
    
    # 4. Embeddings are implicitly initialized to 0 by np.zeros

    return vertex_data, edge_data, connectivity

if __name__ == "__main__":
    print("--- Testing CNF Feature Extraction ---")

    # Define a simple CNF problem
    # Variables: 1, 2, 3 (total_vars_in_cnf = 3)
    # Clauses: (x1 V !x2), (x2 V x3)
    # In solver format: [[1, -2], [2, 3]] (1-indexed literals)
    test_total_vars = 3
    test_clauses = [[1, -2], [2, 3]]
    
    # Define embedding dimensions (can be changed as needed)
    test_var_embed_dim = 16 
    test_clause_embed_dim = 16

    print(f"CNF: {test_total_vars} variables, Clauses: {test_clauses}")
    print(f"Var Embedding Dim: {test_var_embed_dim}, Clause Embedding Dim: {test_clause_embed_dim}")

    v_data, e_data, conn_data = generate_graph_features_for_cnf(
        test_total_vars, 
        test_clauses, 
        test_var_embed_dim, 
        test_clause_embed_dim
    )

    print("\n--- Vertex Data (vertex_data) ---")
    print(f"Shape: {v_data.shape}")
    print("Content (Node Type, Original ID, Handcrafted Features ..., Embedding Space ...):")
    # Print variable nodes
    print("Variable Nodes:")
    for i in range(test_total_vars):
        print(f"  Var Node {i} (Original ID {int(v_data[i, NODE_ID_COL])}): {v_data[i, :HANDCRAFTED_FEATURES_START_COL + NUM_HANDCRAFTED_VAR_FEATURES + 3]}... (remaining are embeddings)") 
        # Showing first few embedding values for brevity

    # Print clause nodes
    print("\nClause Nodes:")
    for i in range(len(test_clauses)):
        clause_node_actual_idx = test_total_vars + i
        print(f"  Clause Node {i} (Original ID {int(v_data[clause_node_actual_idx, NODE_ID_COL])}): {v_data[clause_node_actual_idx, :HANDCRAFTED_FEATURES_START_COL + NUM_HANDCRAFTED_CLAUSE_FEATURES + 3]}... (remaining are embeddings)")
        # Showing first few embedding values for brevity
    
    # For full data if needed:
    # print(v_data)

    print("\n--- Edge Data (edge_data) ---")
    print(f"Shape: {e_data.shape}")
    print("Content (Features for each edge pair - [pos_lit_feat, neg_lit_feat]):")
    print(e_data)

    print("\n--- Connectivity Data (connectivity) ---")
    print(f"Shape: {conn_data.shape}")
    print("Content (Source node index, Target node index for each edge):")
    print(conn_data)

    print("\n--- Verification Notes ---")
    print(f"Expected Vertex Data Size: (NumVars + NumClauses, MaxFeatureDim)")
    print(f"  ({test_total_vars} + {len(test_clauses)}, {v_data.shape[1]}) = ({test_total_vars + len(test_clauses)}, {v_data.shape[1]})")
    print(f"Expected Edge Data Size: (TotalLiteralsInClauses * 2, {EDGE_IN_SIZE})")
    num_lits = sum(len(c) for c in test_clauses)
    print(f"  ({num_lits} * 2, {EDGE_IN_SIZE}) = ({num_lits * 2}, {EDGE_IN_SIZE})")
    print(f"Connectivity should have shape (2, TotalLiteralsInClauses * 2)")

    print("\nFeature structure for a Variable Node (example, first few handcrafted + first few embedding):")
    print(f"[ {NODE_TYPE_VAR} (type), OriginalVarID (id), HF1, HF2, HF3, HF4, HF5, Emb1, Emb2, Emb3, ... ]")
    print("Feature structure for a Clause Node (example, first few handcrafted + first few embedding):")
    print(f"[ {NODE_TYPE_CLAUSE} (type), OriginalClauseID (id), HF1, ..., HF15, Emb1, Emb2, Emb3, ... ]")

    print("\nTest script finished.")

