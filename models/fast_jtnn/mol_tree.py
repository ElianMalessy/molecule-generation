from typing import List, Optional, Tuple

import rdkit
import rdkit.Chem as Chem

from models.fast_jtnn.chemutils import (
    get_clique_mol, tree_decomp, get_mol, 
    get_smiles, set_atommap, enum_assemble
)


class MolTreeNode:
    """
    Node for the Junction Tree representing a molecular clique.
    """
    def __init__(self, smiles: str, clique: Optional[List[int]] = None):
        self.smiles: str = smiles
        self.mol: Chem.Mol = get_mol(self.smiles)

        # Avoid mutable default arguments; safely copy the clique
        self.clique: List[int] = list(clique) if clique is not None else []
        self.neighbors: List['MolTreeNode'] = []
        
        # Attributes populated later by the MolTree or assembly process
        self.is_leaf: bool = False
        self.nid: int = 0
        self.idx: int = 0
        self.label: str = ""
        self.cands: List[str] = []

    def add_neighbor(self, nei_node: 'MolTreeNode') -> None:
        self.neighbors.append(nei_node)

    def recover(self, original_mol: Chem.Mol) -> str:
        """
        Recovers the label of the tree node by masking the appropriate atoms.
        """
        clique_atoms = list(self.clique)
        
        if not self.is_leaf:
            for cidx in self.clique:
                original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(self.nid)

        for nei_node in self.neighbors:
            clique_atoms.extend(nei_node.clique)
            if nei_node.is_leaf:  # Leaf node, no need to mark
                continue
            
            for cidx in nei_node.clique:
                # Allow singleton node to override the atom mapping
                if cidx not in self.clique or len(nei_node.clique) == 1:
                    atom = original_mol.GetAtomWithIdx(cidx)
                    atom.SetAtomMapNum(nei_node.nid)

        clique_atoms = list(set(clique_atoms))
        label_mol = get_clique_mol(original_mol, clique_atoms)
        
        # Parse the sub-molecule smiles and ensure it's valid to fix typing errors
        parsed_mol = Chem.MolFromSmiles(get_smiles(label_mol))
        if parsed_mol is None:
            raise ValueError(f"Failed to parse SMILES for clique molecule.")
            
        self.label = Chem.MolToSmiles(parsed_mol)

        # Reset atom mappings for the original molecule
        for cidx in clique_atoms:
            original_mol.GetAtomWithIdx(cidx).SetAtomMapNum(0)

        return self.label
    
    def assemble(self) -> None:
        """
        Enumerates possible attachments of neighboring cliques.
        """
        # Separate neighbors into singletons and complex rings/fragments
        complex_neighbors = sorted(
            [nei for nei in self.neighbors if nei.mol.GetNumAtoms() > 1], 
            key=lambda x: x.mol.GetNumAtoms(), 
            reverse=True
        )
        singletons = [nei for nei in self.neighbors if nei.mol.GetNumAtoms() == 1]
        
        # Evaluate singletons first
        ordered_neighbors = singletons + complex_neighbors

        cands, aroma = enum_assemble(self, ordered_neighbors)
        
        # Filter candidates by aroma score
        valid_cands = [cand for cand, arom in zip(cands, aroma) if arom >= 0]

        if valid_cands:
            # Unzip to extract just the SMILES strings (index 0 of the tuple)
            self.cands = list(list(zip(*valid_cands))[0])
        else:
            self.cands = []


class MolTree:
    """
    Junction Tree representation of a molecule.
    """
    def __init__(self, smiles: str):
        self.smiles: str = smiles
        self.mol: Chem.Mol = get_mol(smiles)

        cliques, edges = tree_decomp(self.mol)
        self.nodes: List[MolTreeNode] = []
        root = 0
        
        # Build nodes
        for i, c in enumerate(cliques):
            cmol = get_clique_mol(self.mol, c)
            node = MolTreeNode(get_smiles(cmol), c)
