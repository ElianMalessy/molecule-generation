import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from tqdm import tqdm
import pandas as pd

class MosesPyGDataset(InMemoryDataset):
    def __init__(self, root, split='train', max_atoms=38, transform=None, pre_transform=None):
        self.split = split
        self.max_atoms = max_atoms
        super().__init__(root, transform, pre_transform)
        # Load safely
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'moses_{self.split}_max{self.max_atoms}.pt']

    def download(self):
        pass

    def process(self):
        # Mapping MOSES heavy atoms -> Dense 0-indexed integers
        # C, N, O, F, S, Cl, Br, H
        atom_map = {6: 0, 7: 1, 8: 2, 9: 3, 16: 4, 17: 5, 35: 6, 1: 7}
        
        url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{self.split}.csv"
        try:
            df = pd.read_csv(url)
        except Exception as e:
            raise RuntimeError(f"Failed to download MOSES data: {e}")
            
        col = next((c for c in df.columns if c.lower() == 'smiles'), df.columns[0])
        smiles_list = df[col].tolist()
            
        data_list = []
        for smiles in tqdm(smiles_list, desc=f"Processing MOSES {self.split}"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None or mol.GetNumAtoms() > self.max_atoms:
                continue

            # Kekulize to get explicit single/double bond types, and clear aromatic flags to avoid confusion
            Chem.Kekulize(mol, clearAromaticFlags=True)

            # Renumber atoms into canonical (Morgan-invariant) order so the same molecule
            # always serializes identically regardless of the SMILES atom ordering.
            ranks = Chem.CanonicalRankAtoms(mol)
            mol = Chem.RenumberAtoms(mol, sorted(range(mol.GetNumAtoms()), key=lambda i: ranks[i]))

            xs = [[atom_map.get(atom.GetAtomicNum(), 0)] for atom in mol.GetAtoms()]
            x = torch.tensor(xs, dtype=torch.long)
            
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds(): # type: ignore
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                bt = bond.GetBondType()
                b_idx = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1, 
                         Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}.get(bt, 0)
                    
                edge_indices.extend([[i, j], [j, i]])
                edge_attrs.extend([[b_idx], [b_idx]])
                
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.tensor(edge_attrs, dtype=torch.long) if edge_attrs else torch.empty((0, 1), dtype=torch.long)
                
            data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                  smiles=Chem.MolToSmiles(mol, isomericSmiles=False)))
            
        self.save(data_list, self.processed_paths[0])
