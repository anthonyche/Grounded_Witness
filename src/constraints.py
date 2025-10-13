# -*- coding: utf-8 -*-
"""
constraints.py
--------------
This module ONLY defines dataset-specific constraints (TGDs) and light-weight
utilities (registry, validation). Pattern matching / backchase logic lives in
another module (e.g., matcher.py) to keep separation of concerns.
"""

from typing import Dict, List, Tuple, Any

# MUTAG node feature order in TUDataset:
# ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']
LABEL_ID: Dict[str, int] = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'I': 4, 'Cl': 5, 'Br': 6
}

# Type aliases (purely documentary; we keep TGDs as dicts by design)
Edge = Tuple[str, str]
NodeSpec = Dict[str, Any]
Pattern = Dict[str, Any]
TGD = Dict[str, Any]


def validate_tgd(tgd: TGD) -> None:
    """
    Sanity-check a TGD dictionary. Raises AssertionError on bad shape.
    We intentionally allow the body to introduce new nodes that do not
    appear in the head (since backchase can insert them).
    """
    assert 'name' in tgd and isinstance(tgd['name'], str)
    assert 'head' in tgd and 'body' in tgd
    for part in ('head', 'body'):
        p = tgd[part]
        assert 'nodes' in p and 'edges' in p
        assert isinstance(p['nodes'], dict), "nodes must be a dict of var->spec"
        assert isinstance(p['edges'], list), "edges must be a list of (u,v)"
        # Optional key
        if 'distinct' in p:
            assert isinstance(p['distinct'], list)
            # All variables in distinct must appear in declared nodes:
            missing = [v for v in p['distinct'] if v not in p['nodes']]
            assert not missing, f"distinct vars not in {part}.nodes: {missing}"

    # Check that all edge endpoints are declared in that part's nodes:
    for part in ('head', 'body'):
        p = tgd[part]
        declared = set(p['nodes'].keys())
        for (u, v) in p['edges']:
            assert u in declared and v in declared, \
                f"edge ({u},{v}) uses undeclared vars in {part}"


# ----------------------------------------------------------------------
# constraints for MUTAG
# ----------------------------------------------------------------------
TGD_C6_CLOSURE_1 = {
  "name": "c6_closure",
  "body": {   # 需要补全或验证的部分（修复目标）
    "nodes": {
      "A": {"in": [LABEL_ID['C']]},
      "D": {"in": [LABEL_ID['C']]},
      "E": {"in": [LABEL_ID['C']]},
      "F": {"in": [LABEL_ID['C']]}
    },
    "edges": [("D","E"),("E","F"),("F","A")],
    "distinct": ["A","D","E","F"]
  },
  "head": {   # 已观测到的部分（触发条件）
    "nodes": {
      "A": {"in": [LABEL_ID['C']]},
      "B": {"in": [LABEL_ID['C']]},
      "C": {"in": [LABEL_ID['C']]},
      "D": {"in": [LABEL_ID['C']]}
    },
    "edges": [("A","B"), ("B","C"), ("C","D")],
    "distinct": ["A","B","C","D"]
  }
}

TGD_C6_CLOSURE_2 = {
    "name": "c6_closure_2",
    "body": {   # 需要补全或验证的部分
        "nodes": {
        "A": {"in": [LABEL_ID['C']]},
        "C": {"in": [LABEL_ID['C']]},
        "D": {"in": [LABEL_ID['C']]},
        "E": {"in": [LABEL_ID['C']]},
        "F": {"in": [LABEL_ID['C']]}
        },
        "edges": [("A","F"),("F","E"),("E","D"),("D","C")],
        "distinct": ["A","D","E","F","C"]
    },
    "head": {   # 已观测到的部分（触发条件）
        "nodes": {
        "A": {"in": [LABEL_ID['C']]},
        "B": {"in": [LABEL_ID['C']]},
        "C": {"in": [LABEL_ID['C']]},
        },
        "edges": [("A","B"), ("B","C")],
        "distinct": ["A","B","C"]
    }
}    

TGD_C6_CLOSURE_3 = {
    "name": "c6_closure_3",
    "body": {   # 需要补全或验证的部分
        "nodes": {
        "A": {"in": [LABEL_ID['C']]},
        "B": {"in": [LABEL_ID['C']]},
        "C": {"in": [LABEL_ID['C']]},
        "G": {"in": [LABEL_ID['C']]}
        },
        "edges": [("A","B"), ("B","C"),("B","G")],
        "distinct": ["A","B","C","G"]
    },
    "head": {   # 已观测到的部分（触发条件）
        "nodes": {
        "A": {"in": [LABEL_ID['C']]},
        "C": {"in": [LABEL_ID['C']]},
        "D": {"in": [LABEL_ID['C']]},
        "E": {"in": [LABEL_ID['C']]},
        "F": {"in": [LABEL_ID['C']]}
        },
        "edges": [("A","F"),("F","E"),("E","D"),("D","C")],
        "distinct": ["A","D","E","F","C"]
    }
}

TGD_NITRO_ON_AROMATIC = {
    "name": "nitro_on_aromatic_completion",
    "body": {   
        "nodes": {
            "C": {"in": [LABEL_ID['C']]},
            "N": {"in": [LABEL_ID['N']]},
            "O1": {"in": [LABEL_ID['O']]},
        },
        "edges": [("C","N"), ("N","O1")],
        "distinct": ["C", "N", "O1"]
    },
    "head": {   
        "nodes": {
            "N": {"in": [LABEL_ID['N']]},
            "O2": {"in": [LABEL_ID['O']]},
        },
        "edges": [("N","O2")],
        "distinct": [ "N", "O2"]
    }
}
TGD_NITRO_ON_NITRO = {
    "name": "nitro_on_nitro_completion",
    "body": {   
        "nodes": {
            "N": {"in": [LABEL_ID['N']]},
            "O1": {"in": [LABEL_ID['O']]},
        },
        "edges": [("N","O1")],
        "distinct": ["N", "O1"]
    },
    "head": {   
        "nodes": {
            "N": {"in": [LABEL_ID['N']]},
            "O2": {"in": [LABEL_ID['O']]},
        },
        "edges": [("N","O2")],
        "distinct": [ "N", "O2"]
    }
}


TGD_C_DOUBLE_O_LIKE = {
    "name": "di-oxy_on_Carbon",
    "body": {   
        "nodes": {
            "C": {"in": [LABEL_ID['C']]},
            "O1": {"in": [LABEL_ID['O']]},
        },
        "edges": [("C","O1")],
        "distinct": ["C", "O1"]
    },
    "head": {  
        "nodes": {
            "C": {"in": [LABEL_ID['C']]},
            "O2": {"in": [LABEL_ID['O']]},
        },
        "edges": [("C","O2")],
        "distinct": ["C", "O2"]
    }
}

TGD_ETHER_LIKE = {
    "name": "ether_like_completion",
    "body": {   # C-O-C 结构,C-O已有
        "nodes": {
            "C1": {"in": [LABEL_ID['C']]},
            "O": {"in": [LABEL_ID['O']]},
        },
        "edges": [("C1","O")],
        "distinct": ["C1", "O"]
    },
    "head": {   # O-C2
        "nodes": {
            "O": {"in": [LABEL_ID['O']]},
            "C2": {"in": [LABEL_ID['C']]},
        },
        "edges": [("O","C2")],
        "distinct": ["O", "C2"]
    },
}

TGD_HALOGEN_ANCHOR = {
    "name": "halogen_anchor_completion",
    "body": {   # C-X 结构, X为卤素
        "nodes": {
            "C":{"in":[LABEL_ID['C']]}, 
            "X":{"in":[LABEL_ID['Cl'], LABEL_ID['Br'], LABEL_ID['F'], LABEL_ID['I']]}
        },
        "edges": [("C","X")],
        "distinct": ["C","X"]
    },
    "head": {   # C-C 卤代碳链回碳骨架
        "nodes": {
            "C":{"in":[LABEL_ID['C']]}, 
            "C2":{"in":[LABEL_ID['C']]}
        },
        "edges": [("C","C2")],
        "distinct": ["C", "C2"]
    }
}

TGD_AMINE_DI_C = {
    "name": "amine_di_carbon_completion",  # N–C 已见 ⇒ N–C₂（胺的二价/取代倾向，宽松先验）
    "body": {
        "nodes": {
            "N":{"in":[LABEL_ID['N']]}, 
            "C1":{"in":[LABEL_ID['C']]}
            },
        "edges": [("N","C1")],
        "distinct": ["N","C1"]
    },
    "head": {
        "nodes": {
            "N":{"in":[LABEL_ID['N']]}, 
            "C2":{"in":[LABEL_ID['C']]}
        },
        "edges": [("N","C2")],
        "distinct": ["N","C2"]
    }
}


# Validate and register constraints for MUTAG.
# Add more TGDs to this list as you design them.
CONSTRAINTS_MUTAG: List[TGD] = []
validate_tgd(TGD_C6_CLOSURE_1)
CONSTRAINTS_MUTAG.append(TGD_C6_CLOSURE_1)
validate_tgd(TGD_C6_CLOSURE_2)
CONSTRAINTS_MUTAG.append(TGD_C6_CLOSURE_2)
validate_tgd(TGD_C6_CLOSURE_3)
CONSTRAINTS_MUTAG.append(TGD_C6_CLOSURE_3)
validate_tgd(TGD_NITRO_ON_AROMATIC)
CONSTRAINTS_MUTAG.append(TGD_NITRO_ON_AROMATIC)
validate_tgd(TGD_C_DOUBLE_O_LIKE)
CONSTRAINTS_MUTAG.append(TGD_C_DOUBLE_O_LIKE)
validate_tgd(TGD_ETHER_LIKE)
CONSTRAINTS_MUTAG.append(TGD_ETHER_LIKE)
validate_tgd(TGD_HALOGEN_ANCHOR)
CONSTRAINTS_MUTAG.append(TGD_HALOGEN_ANCHOR)
validate_tgd(TGD_AMINE_DI_C)
CONSTRAINTS_MUTAG.append(TGD_AMINE_DI_C)  
validate_tgd(TGD_NITRO_ON_NITRO)
CONSTRAINTS_MUTAG.append(TGD_NITRO_ON_NITRO)

# ----------------------------------------------------------------------
# constraints for Yelp (Node Classification)
# ----------------------------------------------------------------------
# Yelp is a user-business review graph with node features but no explicit labels.
# We use KMeans clustering (n_clusters=16) to assign pseudo node types (y_type)
# based on 300-dim node features. This makes TGDs more specific and reduces
# spurious matches compared to "match any node" constraints.
#
# Design principles:
#   1. Use y_type (0-15) instead of y (multi-label class sets)
#   2. Constraints capture structural patterns conditioned on node types
#   3. Triangle/Square patterns work better with type constraints
#
# Node types will be generated in utils.dataset_func() via:
#   data.y_type = KMeans(n_clusters=16).fit_predict(data.x)

# Helper: Use specific types to reduce spurious matches
# Based on KMeans clustering distribution, use most common types
# Type distribution: [15055, 53286, 35267, 108357, 59889, 33919, 37681, 88871, ...]
# Type 3 is most common (108357 nodes), use it for patterns

# TGD 1: Triangle Completion for Type-3 (1-edge HEAD → 2-edge BODY)
# HEAD: If we see A→B edge between type-3 nodes
# BODY: Then there should exist a third node C (type-3) with A→C and C→B (triangle)
TGD_YELP_TRIANGLE_TYPE3 = {
    "name": "yelp_triangle_type3",
    "head": {
        "nodes": {
            "A":{"in": [3]},  # Type 3 (108K nodes, 5 in subgraph)
            "B":{"in": [3]},
        },
        "edges": [("A", "B")],  # Single edge observed
        "distinct": ["A", "B"]
    },
    "body": {
        "nodes": {
            "A":{"in": [3]},
            "B":{"in": [3]},
            "C":{"in": [3]},  # Third node to complete triangle
        },
        "edges": [("A", "C"), ("C", "B")],  # Two edges to form triangle
        "distinct": ["A", "B", "C"]
    }
}


# TGD 2: Triangle Completion for Type-7 (1-edge HEAD → 2-edge BODY)
TGD_YELP_TRIANGLE_TYPE7 = {
    "name": "yelp_triangle_type7",
    "head": {
        "nodes": {
            "A":{"in": [7]},  # Type 7 (88K nodes, 3 in subgraph)
            "B":{"in": [7]},
        },
        "edges": [("A", "B")],
        "distinct": ["A", "B"]
    },
    "body": {
        "nodes": {
            "A":{"in": [7]},
            "B":{"in": [7]},
            "C":{"in": [7]},
        },
        "edges": [("A", "C"), ("C", "B")],
        "distinct": ["A", "B", "C"]
    }
}

# TGD 3: Triangle Completion for Type-1 (1-edge HEAD → 2-edge BODY)
TGD_YELP_TRIANGLE_TYPE1 = {
    "name": "yelp_triangle_type1",
    "head": {
        "nodes": {
            "A":{"in": [1]},  # Type 1 (53K nodes)
            "B":{"in": [1]},
        },
        "edges": [("A", "B")],
        "distinct": ["A", "B"]
    },
    "body": {
        "nodes": {
            "A":{"in": [1]},
            "B":{"in": [1]},
            "C":{"in": [1]},
        },
        "edges": [("A", "C"), ("C", "B")],
        "distinct": ["A", "B", "C"]
    }
}

# TGD 4: Triangle Completion for Type-4 (1-edge HEAD → 2-edge BODY)
TGD_YELP_TRIANGLE_TYPE4 = {
    "name": "yelp_triangle_type4",
    "head": {
        "nodes": {
            "A":{"in": [4]},  # Type 4 (59K nodes)
            "B":{"in": [4]},
        },
        "edges": [("A", "B")],
        "distinct": ["A", "B"]
    },
    "body": {
        "nodes": {
            "A":{"in": [4]},
            "B":{"in": [4]},
            "C":{"in": [4]},
        },
        "edges": [("A", "C"), ("C", "B")],
        "distinct": ["A", "B", "C"]
    }
}


# TGD 3: Cross-Type Bridge Patterns (Multiple Type Combinations)
# Based on type distribution: [15055, 53286, 35267, 108357, 59889, 33919, 37681, 88871, ...]
# Use common types: 1(53K), 3(108K), 4(59K), 7(88K)

# Bridge 1: Type-1 nodes via Type-4 bridge
TGD_YELP_BRIDGE_1_VIA_4 = {
    "name": "yelp_bridge_type1_via_type4",
    "head": {
        "nodes": {
            "A1":{"in": [1]},  # Type 1 (53K nodes)
            "B": {"in": [4]},  # Type 4 bridge (59K nodes)
            "A2":{"in": [1]},
        },
        "edges": [("A1", "B"), ("B", "A2")],
        "distinct": ["A1", "B", "A2"]
    },
    "body": {
        "nodes": {
            "A1":{"in": [1]},
            "A2":{"in": [1]},
        },
        "edges": [("A1", "A2")],
        "distinct": ["A1", "A2"]
    }
}

# Bridge 2: Type-3 nodes via Type-7 bridge
TGD_YELP_BRIDGE_3_VIA_7 = {
    "name": "yelp_bridge_type3_via_type7",
    "head": {
        "nodes": {
            "A1":{"in": [3]},  # Type 3 (108K nodes - most common)
            "B": {"in": [7]},  # Type 7 bridge (88K nodes)
            "A2":{"in": [3]},
        },
        "edges": [("A1", "B"), ("B", "A2")],
        "distinct": ["A1", "B", "A2"]
    },
    "body": {
        "nodes": {
            "A1":{"in": [3]},
            "A2":{"in": [3]},
        },
        "edges": [("A1", "A2")],
        "distinct": ["A1", "A2"]
    }
}

# Bridge 3: Type-4 nodes via Type-1 bridge (reverse of Bridge 1)
TGD_YELP_BRIDGE_4_VIA_1 = {
    "name": "yelp_bridge_type4_via_type1",
    "head": {
        "nodes": {
            "A1":{"in": [4]},  # Type 4
            "B": {"in": [1]},  # Type 1 bridge
            "A2":{"in": [4]},
        },
        "edges": [("A1", "B"), ("B", "A2")],
        "distinct": ["A1", "B", "A2"]
    },
    "body": {
        "nodes": {
            "A1":{"in": [4]},
            "A2":{"in": [4]},
        },
        "edges": [("A1", "A2")],
        "distinct": ["A1", "A2"]
    }
}

# Bridge 4: Type-0 nodes via Type-3 bridge (smaller type via common type)
TGD_YELP_BRIDGE_0_VIA_3 = {
    "name": "yelp_bridge_type0_via_type3",
    "head": {
        "nodes": {
            "A1":{"in": [0]},  # Type 0 (15K nodes)
            "B": {"in": [3]},  # Type 3 bridge (108K - most common)
            "A2":{"in": [0]},
        },
        "edges": [("A1", "B"), ("B", "A2")],
        "distinct": ["A1", "B", "A2"]
    },
    "body": {
        "nodes": {
            "A1":{"in": [0]},
            "A2":{"in": [0]},
        },
        "edges": [("A1", "A2")],
        "distinct": ["A1", "A2"]
    }
}

# Bridge 5: Type-2 nodes via Type-6 bridge (mid-size types)
TGD_YELP_BRIDGE_2_VIA_6 = {
    "name": "yelp_bridge_type2_via_type6",
    "head": {
        "nodes": {
            "A1":{"in": [2]},  # Type 2 (35K nodes)
            "B": {"in": [6]},  # Type 6 bridge (37K nodes)
            "A2":{"in": [2]},
        },
        "edges": [("A1", "B"), ("B", "A2")],
        "distinct": ["A1", "B", "A2"]
    },
    "body": {
        "nodes": {
            "A1":{"in": [2]},
            "A2":{"in": [2]},
        },
        "edges": [("A1", "A2")],
        "distinct": ["A1", "A2"]
    }
}

# Validate and register Yelp constraints
# Triangle (1 variant) + Square + Bridges (5 variants) = 7 total
CONSTRAINTS_YELP: List[TGD] = []

# Triangle completion constraints (1-edge HEAD → 2-edge BODY)
validate_tgd(TGD_YELP_TRIANGLE_TYPE3)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_TYPE3)
validate_tgd(TGD_YELP_TRIANGLE_TYPE7)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_TYPE7)
validate_tgd(TGD_YELP_TRIANGLE_TYPE1)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_TYPE1)
validate_tgd(TGD_YELP_TRIANGLE_TYPE4)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_TYPE4)

# Keep bridge patterns (cross-type, 2-edge HEAD)
validate_tgd(TGD_YELP_BRIDGE_1_VIA_4)
CONSTRAINTS_YELP.append(TGD_YELP_BRIDGE_1_VIA_4)
validate_tgd(TGD_YELP_BRIDGE_3_VIA_7)
CONSTRAINTS_YELP.append(TGD_YELP_BRIDGE_3_VIA_7)
validate_tgd(TGD_YELP_BRIDGE_4_VIA_1)
CONSTRAINTS_YELP.append(TGD_YELP_BRIDGE_4_VIA_1)
validate_tgd(TGD_YELP_BRIDGE_0_VIA_3)
CONSTRAINTS_YELP.append(TGD_YELP_BRIDGE_0_VIA_3)
validate_tgd(TGD_YELP_BRIDGE_2_VIA_6)
CONSTRAINTS_YELP.append(TGD_YELP_BRIDGE_2_VIA_6)


# ----------------------------------------------------------------------
# Constraints for Cora (Citation Network - Node Classification)
# ----------------------------------------------------------------------
# Cora is a citation network with 7 paper categories:
#   0: Case_Based
#   1: Genetic_Algorithms  
#   2: Neural_Networks
#   3: Probabilistic_Methods
#   4: Reinforcement_Learning
#   5: Rule_Learning
#   6: Theory
#
# Citation patterns in academic networks:
#   1. Papers in the same field cite each other (homophily)
#   2. Review papers cite many papers in the field
#   3. Transitivity: if A cites B, and B cites C, likely A and C are related
#
# Design principles:
#   - Use actual node labels (y) instead of synthetic types
#   - Capture citation homophily (same-class citations)
#   - Use directed edges (A→B means A cites B)

# TGD 1: Citation Homophily - Same Class Co-citation
# If paper A and B (same class i) both cite paper C (same class i), 
# then A and B likely cite each other (they work on related topics)
TGD_CORA_CITATION_TRIANGLE = {
    "name": "cora_citation_triangle",
    "head": {
        "nodes": {
            "A": {"in": list(range(7))},  # Any class
            "B": {"in": list(range(7))},  # Same class as A
            "C": {"in": list(range(7))},  # Same class as A and B
        },
        "edges": [("A", "C"), ("B", "C")],  # A and B both cite C (co-citation)
        "distinct": ["A", "B", "C"]
    },
    "body": {
        "nodes": {
            "A": {"in": list(range(7))},
            "B": {"in": list(range(7))},
            "C": {"in": list(range(7))},
        },
        "edges": [("A", "B")],  # Then A cites B (or B cites A)
        "distinct": ["A", "B", "C"]
    }
}

# TGD 2: Cross-Field Citation (NN ↔ RL)
# If NN paper A and RL paper B both cite bridge paper C, then A and B are related
TGD_CORA_NN_RL_BRIDGE = {
    "name": "cora_nn_rl_citation",
    "head": {
        "nodes": {
            "A": {"in": [2]},  # Neural Networks paper
            "B": {"in": [4]},  # Reinforcement Learning paper
            "C": {"in": [2, 4]},  # Bridge paper (NN or RL)
        },
        "edges": [("A", "C"), ("B", "C")],  # Both cite C (co-citation)
        "distinct": ["A", "B", "C"]
    },
    "body": {
        "nodes": {
            "A": {"in": [2]},
            "B": {"in": [4]},
            "C": {"in": [2, 4]},
        },
        "edges": [("A", "B")],  # Then A cites B (cross-field citation)
        "distinct": ["A", "B", "C"]
    }
}

# TGD 3: Theory Papers as Hubs
# If two applied papers A and B both cite the same Theory paper T, then A and B are related
TGD_CORA_THEORY_HUB = {
    "name": "cora_theory_hub",
    "head": {
        "nodes": {
            "T": {"in": [6]},  # Theory paper (hub)
            "A": {"in": [0, 1, 2, 3, 4, 5]},  # Applied paper 1 (not Theory)
            "B": {"in": [0, 1, 2, 3, 4, 5]},  # Applied paper 2 (not Theory)
        },
        "edges": [("A", "T"), ("B", "T")],  # Both cite Theory (co-citation)
        "distinct": ["T", "A", "B"]
    },
    "body": {
        "nodes": {
            "T": {"in": [6]},
            "A": {"in": [0, 1, 2, 3, 4, 5]},
            "B": {"in": [0, 1, 2, 3, 4, 5]},
        },
        "edges": [("A", "B")],  # Then A cites B (or vice versa)
        "distinct": ["T", "A", "B"]
    }
}

# TGD 4: Probabilistic Methods ↔ Neural Networks
# If Prob and NN papers both cite a bridge paper, they're related (Bayesian NNs, etc.)
TGD_CORA_PROB_NN_BRIDGE = {
    "name": "cora_prob_nn_citation",
    "head": {
        "nodes": {
            "P": {"in": [3]},  # Probabilistic Methods
            "N": {"in": [2]},  # Neural Networks
            "C": {"in": [2, 3]},  # Bridge paper (NN or Prob)
        },
        "edges": [("P", "C"), ("N", "C")],  # Both cite C (co-citation)
        "distinct": ["P", "N", "C"]
    },
    "body": {
        "nodes": {
            "P": {"in": [3]},
            "N": {"in": [2]},
            "C": {"in": [2, 3]},
        },
        "edges": [("P", "N")],  # Then P cites N (cross-field citation)
        "distinct": ["P", "N", "C"]
    }
}

# TGD 5: Genetic Algorithms ↔ Rule Learning
# If GA and RL papers both cite a bridge paper, they're related (symbolic approaches)
TGD_CORA_GA_RULE_BRIDGE = {
    "name": "cora_ga_rule_citation",
    "head": {
        "nodes": {
            "G": {"in": [1]},  # Genetic Algorithms
            "R": {"in": [5]},  # Rule Learning
            "C": {"in": [1, 5]},  # Bridge paper (GA or Rule)
        },
        "edges": [("G", "C"), ("R", "C")],  # Both cite C (co-citation)
        "distinct": ["G", "R", "C"]
    },
    "body": {
        "nodes": {
            "G": {"in": [1]},
            "R": {"in": [5]},
            "C": {"in": [1, 5]},
        },
        "edges": [("G", "R")],  # Then G cites R (cross-field citation)
        "distinct": ["G", "R", "C"]
    }
}

# Validate and register Cora constraints
CONSTRAINTS_CORA: List[TGD] = []
validate_tgd(TGD_CORA_CITATION_TRIANGLE)
CONSTRAINTS_CORA.append(TGD_CORA_CITATION_TRIANGLE)
validate_tgd(TGD_CORA_NN_RL_BRIDGE)
CONSTRAINTS_CORA.append(TGD_CORA_NN_RL_BRIDGE)
validate_tgd(TGD_CORA_THEORY_HUB)
CONSTRAINTS_CORA.append(TGD_CORA_THEORY_HUB)
validate_tgd(TGD_CORA_PROB_NN_BRIDGE)
CONSTRAINTS_CORA.append(TGD_CORA_PROB_NN_BRIDGE)
validate_tgd(TGD_CORA_GA_RULE_BRIDGE)
CONSTRAINTS_CORA.append(TGD_CORA_GA_RULE_BRIDGE)


# Central registry by dataset key.
_REGISTRY: Dict[str, List[TGD]] = {
    'MUTAG': CONSTRAINTS_MUTAG,
    'YELP': CONSTRAINTS_YELP,
    'CORA': CONSTRAINTS_CORA,
    # 'BASHAPE': [...],   # placeholder for future constraints
    # 'ATLAS': [...],
}


def get_constraints(dataset_key: str) -> List[TGD]:
    """
    Return a list of TGDs for a given dataset key (e.g., 'MUTAG').
    Caller can further filter/extend the list via config.
    """
    key = dataset_key.upper()
    if key not in _REGISTRY:
        # Return an empty list instead of raising, so the caller can proceed.
        return []
    return _REGISTRY[key]


__all__ = [
    'LABEL_ID',
    'TGD_C6_CLOSURE',
    'CONSTRAINTS_MUTAG',
    'CONSTRAINTS_YELP',
    'CONSTRAINTS_CORA',
    'get_constraints',
]