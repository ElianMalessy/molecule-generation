import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem
from tqdm import tqdm

class MosesPyGDataset(InMemoryDataset):
    """
    Downloads and converts the MOSES SMILES dataset into PyG Graph Data objects.
    Bypasses the 'molsets' dependency by pulling the CSV directly from GitHub.
    Extracts atomic numbers and bond types to match the structure of PyG's ZINC dataset.
    """
    def __init__(self, root, split='train', max_atoms=38, transform=None, pre_transform=None):
        self.split = split
        self.max_atoms = max_atoms
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        # Tie the cache name to max_atoms so it recompiles if you change the config
        return [f'moses_{self.split}_max{self.max_atoms}.pt']

    def download(self):
        pass

    def process(self):
        import pandas as pd
        from rdkit import Chem
        import urllib.request
        
        # MOSES heavy atoms: C, N, S, O, F, Cl, Br, H
        # Map them to dense integers. 8 is a fallback.
        atom_map = {6: 0, 7: 1, 8: 2, 9: 3, 16: 4, 17: 5, 35: 6, 1: 7}
        
        print(f"Downloading MOSES '{self.split}' split directly from GitHub...")
        url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{self.split}.csv"
        
        try:
            df = pd.read_csv(url)
        except Exception as e:
            raise RuntimeError(f"Failed to download MOSES data from {url}. Error: {e}")
            
        # Extract SMILES list (handling potential casing differences in the CSV header)
        if 'SMILES' in df.columns:
            smiles_list = df['SMILES'].tolist()
        elif 'smiles' in df.columns:
            smiles_list = df['smiles'].tolist()
        else:
            smiles_list = df.iloc[:, 0].tolist() # Fallback to the first column
            
        data_list = []
        
        for smiles in tqdm(smiles_list, desc=f"Processing MOSES {self.split}"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Restrict to molecules that fit the maximum size
            if mol.GetNumAtoms() > self.max_atoms:
                continue
                
            # Build Node Features (Atom Types)
            xs = []
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                idx = atom_map.get(atomic_num, 8)
                xs.append([idx])
            
            x = torch.tensor(xs, dtype=torch.long)
            
            # Build Edge Features (Bond Types)
            edge_indices, edge_attrs = [], []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                bt = bond.GetBondType()
                if bt == Chem.BondType.SINGLE:
                    b_idx = 0
                elif bt == Chem.BondType.DOUBLE:
                    b_idx = 1
                elif bt == Chem.BondType.TRIPLE:
                    b_idx = 2
                elif bt == Chem.BondType.AROMATIC:
                    b_idx = 3
                else:
                    b_idx = 0
                    
                # Undirected edges
                edge_indices += [[i, j], [j, i]]
                edge_attrs += [[b_idx], [b_idx]]
                
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_attrs, dtype=torch.long)
            else:
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, 1), dtype=torch.long)
                
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            data_list.append(data)
            
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
