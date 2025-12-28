# @Time   : 2020/7/20
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time   : 2022/7/8, 2020/10/3, 2020/10/1
# @Author : Zhen Tian, Yupeng Hou, Zihan Lin
# @Email  : chenyuwuxinn@gmail.com, houyupeng@ruc.edu.cn, zhlin@ruc.edu.cn

import argparse
import os

from recbole.quick_start import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="BPR", help="name of models")
    parser.add_argument(
        "--dataset", "-d", type=str, default="ml-100k", help="name of datasets"
    )
    parser.add_argument("--config_files", type=str, default=None, help="config files")
    parser.add_argument(
        "--nproc", type=int, default=1, help="the number of process in this group"
    )
    parser.add_argument(
        "--ip", type=str, default="localhost", help="the ip of master node"
    )
    parser.add_argument(
        "--port", type=str, default="5678", help="the port of master node"
    )
    parser.add_argument(
        "--world_size", type=int, default=-1, help="total number of jobs"
    )
    parser.add_argument(
        "--group_offset",
        type=int,
        default=0,
        help="the global rank offset of this group",
    )
    # Score saving for visualization
    parser.add_argument(
        "--save_test_scores", action="store_true", 
        help="save test prediction scores for visualization/analysis"
    )
    parser.add_argument(
        "--scores_output_dir", type=str, default="ablation_study_doc/scores",
        help="directory to save test scores"
    )
    parser.add_argument(
        "--variant_name", type=str, default=None,
        help="variant name for score file naming (default: model_dataset)"
    )

    args, _ = parser.parse_known_args()

    config_file_list = (
        args.config_files.strip().split(" ") if args.config_files else None
    )

    # Build score save path if requested
    save_scores_path = None
    if args.save_test_scores:
        variant_name = args.variant_name or f"{args.model}_{args.dataset}"
        variant_name = variant_name.replace(" ", "_").replace("+", "_").replace(",", "_")
        save_scores_path = os.path.join(args.scores_output_dir, variant_name)

    run(
        args.model,
        args.dataset,
        config_file_list=config_file_list,
        nproc=args.nproc,
        world_size=args.world_size,
        ip=args.ip,
        port=args.port,
        group_offset=args.group_offset,
        save_scores_path=save_scores_path,
    )
