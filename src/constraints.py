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

# Central registry by dataset key.
_REGISTRY: Dict[str, List[TGD]] = {
    'MUTAG': CONSTRAINTS_MUTAG,
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
    'get_constraints',
]