import torch
import torch.nn as nn
import torch.nn.functional as F
import rdkit.Chem as Chem
from typing import List, Tuple, Any

ELEM_LIST = ["C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "Al", "I", "B", "K", "Se", "Zn", "H", "Cu", "Mn", "unknown"]
ATOM_FDIM = len(ELEM_LIST) + 6 + 5 + 1
BOND_FDIM = 5 
MAX_NB = 15

def onek_encoding_unk(x: Any, allowable_set: List[Any]) -> List[int]:
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]

def atom_features(atom: Chem.Atom) -> List[float]:
    # Returns standard python list to be batched into a single tensor later (avoids tensor creation overhead)
    return (onek_encoding_unk(atom.GetSymbol(), ELEM_LIST) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetFormalCharge(), [-1,-2,1,2,0])
            + [int(atom.GetIsAromatic())])

def bond_features(bond: Chem.Bond) -> List[float]:
    bt = bond.GetBondType()
    return [int(bt == Chem.rdchem.BondType.SINGLE), 
            int(bt == Chem.rdchem.BondType.DOUBLE), 
            int(bt == Chem.rdchem.BondType.TRIPLE), 
            int(bt == Chem.rdchem.BondType.AROMATIC), 
            int(bond.IsInRing())]

class JTMPN(nn.Module):
    def __init__(self, hidden_size: int, depth: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth

        self.W_i = nn.Linear(ATOM_FDIM + BOND_FDIM, hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_o = nn.Linear(ATOM_FDIM + hidden_size, hidden_size)
    
    def forward(self, fatoms: torch.Tensor, fbonds: torch.Tensor, agraph: torch.Tensor, bgraph: torch.Tensor, scope: List[Tuple[int, int]], tree_message: torch.Tensor) -> torch.Tensor: 
        # Device mapping happens natively here
        device = self.W_i.weight.device
        fatoms = fatoms.to(device)
        fbonds = fbonds.to(device)
        agraph = agraph.to(device)
        bgraph = bgraph.to(device)
        tree_message = tree_message.to(device)

        binput = self.W_i(fbonds)
        graph_message = F.relu(binput)

        for i in range(self.depth - 1):
            message = torch.cat([tree_message, graph_message], dim=0) 
            # Modern, highly optimized equivalent of the legacy index_select_sum
            nei_message = message[bgraph].sum(dim=1) 
            nei_message = self.W_h(nei_message)
            graph_message = F.relu(binput + nei_message) 

        message = torch.cat([tree_message, graph_message], dim=0)
        
        # Replaces second index_select_sum
        nei_message = message[agraph].sum(dim=1)
        
        ainput = torch.cat([fatoms, nei_message], dim=1)
        atom_hiddens = F.relu(self.W_o(ainput))

        mol_vecs = []
        for st, le in scope:
            mol_vec = atom_hiddens.narrow(0, st, le).sum(dim=0) / le
            mol_vecs.append(mol_vec)

        return torch.stack(mol_vecs, dim=0)

    @staticmethod
    def tensorize(cand_batch: List[Any], mess_dict: dict) -> Tuple:
        fatoms, fbonds = [], [] 
        in_bonds, all_bonds = [], [] 
        total_atoms = 0
        total_mess = len(mess_dict) + 1 
        scope = []

        for smiles, all_nodes, ctr_node in cand_batch:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles}")

            Chem.Kekulize(mol)
            n_atoms = mol.GetNumAtoms()
            ctr_bid = ctr_node.idx

            for atom in mol.GetAtoms():
                fatoms.append(atom_features(atom))
                in_bonds.append([]) 
        
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms
                x_nid, y_nid = a1.GetAtomMapNum(), a2.GetAtomMapNum()
                x_bid = all_nodes[x_nid - 1].idx if x_nid > 0 else -1
                y_bid = all_nodes[y_nid - 1].idx if y_nid > 0 else -1

                bfeature = bond_features(bond)

                # Add forward bond
                b = total_mess + len(all_bonds)  
                all_bonds.append((x,y))
                fbonds.append(fatoms[x] + bfeature) # List concat
                in_bonds[y].append(b)

                # Add reverse bond
                b = total_mess + len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append(fatoms[y] + bfeature)
                in_bonds[x].append(b)

                if x_bid >= 0 and y_bid >= 0 and x_bid != y_bid:
                    if (x_bid,y_bid) in mess_dict:
                        in_bonds[y].append(mess_dict[(x_bid,y_bid)])
                    if (y_bid,x_bid) in mess_dict:
                        in_bonds[x].append(mess_dict[(y_bid,x_bid)])
            
            scope.append((total_atoms, n_atoms))
            total_atoms += n_atoms
        
        total_bonds = len(all_bonds)
        # Optimized single-pass tensor instantiation
        fatoms = torch.tensor(fatoms, dtype=torch.float32)
        fbonds = torch.tensor(fbonds, dtype=torch.float32)
        agraph = torch.zeros(total_atoms, MAX_NB, dtype=torch.long)
        bgraph = torch.zeros(total_bonds, MAX_NB, dtype=torch.long)

        for a in range(total_atoms):
            for i, b in enumerate(in_bonds[a]):
                agraph[a, i] = b

        for b1 in range(total_bonds):
            x, y = all_bonds[b1]
            for i, b2 in enumerate(in_bonds[x]): 
                if b2 < total_mess or all_bonds[b2 - total_mess][0] != y:
                    bgraph[b1, i] = b2

        return (fatoms, fbonds, agraph, bgraph, scope)
