# Copyright (c) Meta Platforms, Inc. and affiliates.

"""
A script to run multinode training with submitit.
"""

import sys
sys.path.append('./')
sys.path.append('./MaskCut')
sys.path.append('./third_party')

import argparse
import os
import uuid
from pathlib import Path

import maskcut_with_submitit as main_func
import submitit
import copy

def parse_args():
    parent_parser = main_func.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MaskCut", parents=[parent_parser])
    parser.add_argument("--ngpus", default=1, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=1, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=1400, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Job dir. Leave empty for automatic.")

    parser.add_argument("--partition", default="learnfair", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')

    # Removed the followings if the main file has it already
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--output_dir', default='',
                    help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--tolerance', default=1, type=int, help='tolerance for finding contours')
    
    return parser.parse_args()


def get_shared_folder() -> Path:
    user = os.getenv("USER")
    if Path("/checkpoint/").is_dir():
        p = Path(f"/checkpoint/{user}/experiments/maskcut/")
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

# Using a for loop for getting the array job and submit all jobs in one single array
class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):

        self._setup_gpu_args()
        main_func.main(self.args)

    def checkpoint(self):
        import os
        import submitit
        from pathlib import Path

        self.args.dist_url = get_init_file().as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = get_shared_folder() / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}
    if args.use_volta32:
        kwargs['slurm_constraint'] = 'volta32gb'
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node, # 40
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=8, # default 8
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="MaskCut")

    # Since it is often necessary to submit over 100 jobs simutanously, 
    # using an array to submit these jobs is a more efficient way.
    args.dist_url = get_init_file().as_uri()
    args.output_dir = args.job_dir
    print(args.output_dir)

    # list_folders = list(range(0, 500))
    end_idx = (1000 - args.job_index) // args.num_folder_per_job + 1
    list_folders = list(range(args.job_index, end_idx))
    jobs = []
    args_list = []
    for folder_index in list_folders:
        args_copy = copy.deepcopy(args)
        args_copy.job_index = folder_index
        args_list.append(args_copy)

    with executor.batch():
        for args in args_list:
            trainer = Trainer(args)
            job = executor.submit(trainer)
            jobs.append(job)
    for job in jobs:
        print("Submitted job_id:", job.job_id)


if __name__ == "__main__":
    main()