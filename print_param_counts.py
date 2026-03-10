import os

import torch

ckpt_root = './checkpoints'

results = []

def main():

    for dataset in sorted(os.listdir(ckpt_root)):
        dataset_path = os.path.join(ckpt_root, dataset)
        if not os.path.isdir(dataset_path):
            continue
        for model_type in sorted(os.listdir(dataset_path)):
            model_type_path = os.path.join(dataset_path, model_type)
            if not os.path.isdir(model_type_path):
                continue
            for variant_folder in sorted(os.listdir(model_type_path)):
                variant_path = os.path.join(model_type_path, variant_folder)
                if not os.path.isdir(variant_path):
                    continue
                best_ckpt = os.path.join(variant_path, 'best.pth')
                if not os.path.isfile(best_ckpt):
                    continue
                try:
                    state = torch.load(best_ckpt, map_location='cpu')
                    if isinstance(state, dict):
                        if 'state_dict' in state:
                            state = state['state_dict']
                    n_params = sum(v.numel() for v in state.values() if hasattr(v, 'numel'))
                    # Group sums by top-level prefix (e.g., encoder, decoder, flow, prop_head, fc_*)
                    groups = {}
                    for k, v in state.items():
                        if not hasattr(v, 'numel'):
                            continue
                        top = k.split('.')[0]
                        groups.setdefault(top, 0)
                        groups[top] += v.numel()

                    print(f"\n{variant_path}: {n_params} parameters\nGroup breakdown:")
                    for g, s in sorted(groups.items(), key=lambda x: -x[1]):
                        print(f"  {g}: {s} params")
                    print("Details:")
                    for k, v in state.items():
                        if hasattr(v, 'shape'):
                            print(f"  {k}: {tuple(v.shape)} ({v.numel()} params)")
                        else:
                            print(f"  {k}: {type(v)}")
                    results.append((variant_path, n_params))
                except Exception as e:
                    print(f"Error loading {variant_path}: {e}")

    # Print results in alphabetical order
    for path, n_params in sorted(results, key=lambda x: x[0]):
        print(f"{path}: {n_params} parameters")

if __name__ == '__main__':
    main()
