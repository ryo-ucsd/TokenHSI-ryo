import sys
sys.path.append("./")

import os
import os.path as osp
import yaml
import argparse
from tqdm import tqdm

from tokenhsi.data.data_utils import process_amass_seq_smplx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_cfg", type=str, default=osp.join(osp.dirname(__file__), "../dataset_cfg.yaml"))
    args = parser.parse_args()

    # load yaml
    with open(args.dataset_cfg, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # input/output dirs
    amass_dir = cfg["amass_dir"]
    output_dir = os.path.join(os.path.dirname(__file__), "motions")

    os.makedirs(output_dir, exist_ok=True)

    # selected motions
    candidates = {
        "carry": cfg["motions"]["carry"],
    }

    for skill, data in candidates.items():
        output_dir_skill = os.path.join(output_dir, skill)
        os.makedirs(output_dir_skill, exist_ok=True)

        pbar = tqdm(data)
        for seq in pbar:

            pbar.set_description(seq)

            fname = os.path.join(amass_dir, seq.replace("+__+", "/"))
            output_path = os.path.join(output_dir_skill, seq[:-4], "smplx_params.npy")

            os.makedirs(osp.dirname(output_path), exist_ok=True)
            
            process_amass_seq_smplx(fname, output_path)

        print("Processed {} sequences!".format(len(data)))