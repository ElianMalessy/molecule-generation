import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit.Chem as Chem
from typing import List, Tuple, Any

# Assuming get_mol is available in your codebase
from models.fast_jtnn.chemutils import get_mol

ELEM_LIST = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "Al", "I", "B", "K", "Se", "Zn", "H", "Cu", "Mn", "unknown"]

ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 4 + 1
BOND_FDIM = 5 + 6
MAX_NB = 6

def onek_encoding_unk(x: Any, allowable_set: List[Any]) -> List[bool]:
    """One-hot encoding with an 'unknown' fallback."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom: Chem.Atom) -> torch.Tensor:
    """Extracts node features from an RDKit atom."""
    features = (
        onek_encoding_unk(atom.GetSymbol(), ELEM_LIST)
        + onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])
        + onek_encoding_unk(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
        + onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])
        + [atom.GetIsAromatic()]
    )
    return torch.tensor(features, dtype=torch.float32)

def bond_features(bond: Chem.Bond) -> torch.Tensor:
    """Extracts edge features from an RDKit bond."""
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.IsInRing()
    ]
    fstereo = onek_encoding_unk(stereo, [0, 1, 2, 3, 4, 5])
    return torch.tensor(fbond + fstereo, dtype=torch.float32)


class MPN(nn.Module):
    """Message Passing Neural Network for Molecular Graphs."""
    
    def __init__(self, hidden_size: int, depth: int):
        super(MPN, self).__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)

    def forward(self, fatoms: torch.Tensor, fbonds: torch.Tensor, 
                agraph: torch.Tensor, bgraph: torch.Tensor, 
                scope: List[Tuple[int, int]]) -> torch.Tensor:
        """
        Forward pass for MPN. 
        Note: Tensors should be moved to the appropriate device (e.g., .cuda()) before passing here.
        """
        device = self.W_i.weight.device
        fatoms = fatoms.to(device)
        fbonds = fbonds.to(device)
        agraph = agraph.to(device)
        bgraph = bgraph.to(device)

        binput = self.W_i(fbonds)
        message = F.relu(binput)

        # Message Passing Loop
        for _ in range(self.depth - 1):
            # Native PyTorch advanced indexing replaces custom `index_select_ND`
            # bgraph shape: [Total_Bonds, MAX_NB]
            # message shape: [Total_Bonds, hidden_size]
            # nei_message shape: [Total_Bonds, MAX_NB, hidden_size]
            nei_message = message[bgraph].sum(dim=1)
            nei_message = self.W_h(nei_message)
            message = F.relu(binput + nei_message)

        # Node Aggregation
        # agraph shape: [Total_Atoms, MAX_NB]
        nei_message = message[agraph].sum(dim=1)
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        # Graph-level Readout (Mean Pooling by molecule scope)
        batch_vecs = [atom_hiddens[st : st + le].mean(dim=0) for st, le in scope]
        mol_vecs = torch.stack(batch_vecs, dim=0)
        
        return mol_vecs 

    @staticmethod
    def tensorize(mol_batch: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]]:
        """
        Converts a list of SMILES strings into batched graph tensors.
        """
        padding = torch.zeros(ATOM_FDIM + BOND_FDIM, dtype=torch.float32)
        fatoms, fbonds = [], [padding]  # Bond 0 is reserved for padding
        in_bonds, all_bonds = [], [(-1, -1)]
        scope = []
        total_atoms = 0

        for smiles in mol_batch:
            mol = get_mol(smiles)
            
            # Prevent linter/runtime errors by gracefully skipping invalid molecules
            if mol is None:
                continue
                
            n_atoms = mol.GetNumAtoms()
            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                # Forward edge
                b1 = len(all_bonds) 
                all_bonds.append((x, y))
                fbonds.append(torch.cat([fatoms[x], bond_features(bond)], dim=0))
                in_bonds[y].append(b1)

                # Backward edge
                b2 = len(all_bonds)
                all_bonds.append((y, x))
                fbonds.append(torch.cat([fatoms[y], bond_features(bond)], dim=0))
                in_bonds[x].append(b2)
            
            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms

        total_bonds = len(all_bonds)
        fatoms_tensor = torch.stack(fatoms, dim=0) if fatoms else torch.empty(0, ATOM_FDIM)
        fbonds_tensor = torch.stack(fbonds, dim=0)
        
        agraph = torch.zeros(total_atoms, MAX_NB, dtype=torch.long)
        bgraph = torch.zeros(total_bonds, MAX_NB, dtype=torch.long)

        # Build adjacency matrices
        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                if i < MAX_NB:  # Safety check for highly connected atoms
                    agraph[a, i] = b

        for b1 in range(1, total_bonds):
            x, y = all_bonds[b1]
            i = 0
            for b2 in in_bonds[x]:
                if all_bonds[b2][0] != y:
                    if i < MAX_NB:
                        bgraph[b1, i] = b2
                        i += 1

        return fatoms_tensor, fbonds_tensor, agraph, bgraph, scope
