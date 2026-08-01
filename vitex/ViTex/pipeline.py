import argparse
import os
from utils.inference_utils import get_model, get_dataset, uncond_gen_api, cond_gen_api,continuation_gen_api, get_chord, get_hyperscore, get_x_inp
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder",type=str, default="uncond")
    parser.add_argument("--mode", type=str, default="uncond")
    parser.add_argument("--bar_gen", type=int, default=8)
    parser.add_argument("--num_gen", type=int, default=1)

    parser.add_argument("--chord_scale", type=float, default=0.9)
    parser.add_argument("--hyperscore_scale", type=float, default=1.2)
    parser.add_argument("--chord_idx", type=int, default=-1)
    parser.add_argument("--hyperscore_idx", type=int, default=-1)
    parser.add_argument("--continuation_idx", type=int, default=-1)
    return parser



DEFAULT_CHORD =  ['Em', 'Em', 'Em', 'Em',
        'D', 'D', 'D', 'D',
        'C', 'C', 'D', 'D',
        'G', 'G', 'B7', 'B7',
        'Em', 'Em', 'Em', 'Em',
        'D', 'D', 'D', 'D',
        'C', 'C', 'C', 'C',
        'B7', 'B7', 'B7', 'B7',
        ]

DEFAULT_VITEX = np.zeros((8,8,11,3), dtype=bool)

DEFAULT_VITEX[:,2,4,0] = True
DEFAULT_VITEX[:,2,4,1] = False
DEFAULT_VITEX[:,2,4,2] = True

DEFAULT_VITEX[4:,3,0,0] = False
DEFAULT_VITEX[4:,3,0,1] = True
DEFAULT_VITEX[4:,3,0,2] = True

DEFAULT_VITEX[:4,3,3,0] = False
DEFAULT_VITEX[:4,3,3,1] = True
DEFAULT_VITEX[:4,3,3,2] = False

DEFAULT_VITEX[4:,5,6,0] = True
DEFAULT_VITEX[4:,5,6,1] = False
DEFAULT_VITEX[4:,5,6,2] = True

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    save_folder = args.folder
    os.makedirs(save_folder, exist_ok=True)
    if args.mode == "uncond":
        uncond_gen_api(get_model(), args.num_gen, args.bar_gen, save_folder)
    elif args.mode == "cond":
        dataset = get_dataset()
        chord = get_chord(DEFAULT_CHORD, None, None, save_folder) if args.chord_idx == -1 else get_chord(None, dataset, args.chord_idx, save_folder)
        hyperscore = get_hyperscore(DEFAULT_VITEX, None, None, save_folder) if args.hyperscore_idx == -1 else get_hyperscore(None, dataset, args.hyperscore_idx, save_folder)
        cond_gen_api(get_model(), args.num_gen, args.bar_gen,
                    chord, args.chord_scale,
                    hyperscore, args.hyperscore_scale,
                    save_folder)
    elif args.mode == "continuation":
        if args.continuation_idx == -1:
            raise ValueError("Reference beginning not specified!")
        dataset = get_dataset()
        chord = get_chord(DEFAULT_CHORD, None, None, save_folder) if args.chord_idx == -1 else get_chord(None, dataset, args.chord_idx, save_folder)
        hyperscore = get_hyperscore(DEFAULT_VITEX, None, None, save_folder) if args.hyperscore_idx == -1 else get_hyperscore(None, dataset, args.hyperscore_idx, save_folder)
        x_inp = get_x_inp(dataset, args.continuation_idx, save_folder)
        continuation_gen_api(get_model(), args.num_gen, args.bar_gen,
                    chord, args.chord_scale,
                    hyperscore, args.hyperscore_scale, x_inp,
                    save_folder)
