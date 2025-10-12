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
        "C": {"in": [LABEL_ID['C']]}
        },
        "edges": [("A","B"), ("B","C")],
        "distinct": ["A","B","C"]
    },
    "head": {   # 已观测到的部分（触发条件）
        "nodes": {
        "A": {"in": [LABEL_ID['C']]},
        "C": {"in": [LABEL_ID['C']]},
        "D": {"in": [LABEL_ID['C']]},
        "E": {"in": [LABEL_ID['C']]},
        "F": {"in": [LABEL_ID['C']]}
        },
        "edges": [("A","D"),("D","E"),("E","F"),("F","C")],
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

# ----------------------------------------------------------------------
# constraints for Yelp (Node Classification)
# ----------------------------------------------------------------------
# Yelp is a user-business review graph with node features but no explicit labels.
# We define TGDs based on frequent patterns/motifs in social review networks:
#   1. Triangle closure (social cohesion)
#   2. Star patterns (popular businesses)
#   3. Bipartite patterns (user-business interactions)
# 
# Since Yelp nodes don't have discrete type labels like MUTAG, we use:
#   - Feature-based clustering (if needed in the future)
#   - For now, we use placeholder "any" type: {"in": list(range(100))}
#     (assuming 100 output classes, can be refined)

# Helper: Create "any node" type constraint (matches any node)

# TGD 1: Triangle Closure
# If nodes A-B-C form a path, they should form a triangle (A-C edge)
# This captures social cohesion / transitive relationships
TGD_YELP_TRIANGLE_CLOSURE = {
    "name": "yelp_triangle_closure",
    "body": {   # Observed: A-B-C path
        "nodes": {
            "A":{"in": [0]},  # Match any node (all have label 0)
            "B":{"in": [0]},
            "C":{"in": [0]},
        },
        "edges": [("A", "B"), ("B", "C")],
        "distinct": ["A", "B", "C"]
    },
    "head": {   # Expected: A-C edge should exist
        "nodes": {
            "A":{"in": [0]},
            "C":{"in": [0]},
        },
        "edges": [("A", "C")],
        "distinct": ["A", "C"]
    }
}

# TGD 2: Triangle Completion (Reverse)
# If A-C and B-C edges exist, A-B should also exist
TGD_YELP_TRIANGLE_COMPLETE = {
    "name": "yelp_triangle_complete",
    "body": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
            "C":{"in": [0]},
        },
        "edges": [("A", "C"), ("B", "C")],
        "distinct": ["A", "B", "C"]
    },
    "head": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
        },
        "edges": [("A", "B")],
        "distinct": ["A", "B"]
    }
}

# TGD 3: Square/Cycle Closure
# If nodes form a 3-path A-B-C-D, they should close to a 4-cycle (A-D edge)
TGD_YELP_SQUARE_CLOSURE = {
    "name": "yelp_square_closure",
    "body": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
            "C":{"in": [0]},
            "D":{"in": [0]},
        },
        "edges": [("A", "B"), ("B", "C"), ("C", "D")],
        "distinct": ["A", "B", "C", "D"]
    },
    "head": {
        "nodes": {
            "A":{"in": [0]},
            "D":{"in": [0]},
        },
        "edges": [("A", "D")],
        "distinct": ["A", "D"]
    }
}

# TGD 4: Star Pattern - Hub Node Connectivity
# If a node H connects to A and B, then A and B might also connect (community structure)
TGD_YELP_STAR_TO_CLIQUE = {
    "name": "yelp_star_to_clique",
    "body": {
        "nodes": {
            "H":{"in": [0]},  # Hub node
            "A":{"in": [0]},
            "B":{"in": [0]},
        },
        "edges": [("H", "A"), ("H", "B")],
        "distinct": ["H", "A", "B"]
    },
    "head": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
        },
        "edges": [("A", "B")],
        "distinct": ["A", "B"]
    }
}

# TGD 5: Diamond Pattern Closure
# If A connects to both B and C, and B connects to C, then completing paths
TGD_YELP_DIAMOND_CLOSURE = {
    "name": "yelp_diamond_closure",
    "body": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
            "C":{"in": [0]},
            "D":{"in": [0]},
        },
        "edges": [("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        "distinct": ["A", "B", "C", "D"]
    },
    "head": {
        "nodes": {
            "B":{"in": [0]},
            "C":{"in": [0]},
        },
        "edges": [("B", "C")],
        "distinct": ["B", "C"]
    }
}


# TGD 6: Mutual Neighbor Connection
# If A and B both connect to C, they likely share other common neighbors
TGD_YELP_COMMON_NEIGHBOR = {
    "name": "yelp_common_neighbor",
    "body": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
            "C":{"in": [0]},
        },
        "edges": [("A", "C"), ("B", "C")],
        "distinct": ["A", "B", "C"]
    },
    "head": {
        "nodes": {
            "A":{"in": [0]},
            "B":{"in": [0]},
            "D":{"in": [0]},
        },
        "edges": [("A", "D"), ("B", "D")],
        "distinct": ["A", "B", "D"]
    }
}

# Validate and register Yelp constraints
CONSTRAINTS_YELP: List[TGD] = []
validate_tgd(TGD_YELP_TRIANGLE_CLOSURE)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_CLOSURE)
validate_tgd(TGD_YELP_TRIANGLE_COMPLETE)
CONSTRAINTS_YELP.append(TGD_YELP_TRIANGLE_COMPLETE)
validate_tgd(TGD_YELP_SQUARE_CLOSURE)
CONSTRAINTS_YELP.append(TGD_YELP_SQUARE_CLOSURE)
validate_tgd(TGD_YELP_STAR_TO_CLIQUE)
CONSTRAINTS_YELP.append(TGD_YELP_STAR_TO_CLIQUE)
validate_tgd(TGD_YELP_DIAMOND_CLOSURE)
CONSTRAINTS_YELP.append(TGD_YELP_DIAMOND_CLOSURE)
validate_tgd(TGD_YELP_COMMON_NEIGHBOR)
CONSTRAINTS_YELP.append(TGD_YELP_COMMON_NEIGHBOR)

# Central registry by dataset key.
_REGISTRY: Dict[str, List[TGD]] = {
    'MUTAG': CONSTRAINTS_MUTAG,
    'YELP': CONSTRAINTS_YELP,
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
    'get_constraints',
]