import os
from dataclasses import dataclass
from enum import Enum
import sys
import subprocess
from typing import Optional
from omegaconf import OmegaConf
import math
import itertools
import yaml
import tempfile


class GPUType(Enum):
    A100 = "a100"
    V100 = "v100"


class MailType(Enum):
    BEGIN = "BEGIN"
    END = "END"
    FAIL = "FAIL"
    ALL = "ALL"
    TIME_LIMIT_90 = "TIME_LIMIT_90"
    TIME_LIMIT_80 = "TIME_LIMIT_80"
    TIME_LIMIT_50 = "TIME_LIMIT_50"
    ARRAY_TASKS = "ARRAY_TASKS"


class Partition(Enum):
    STANDARD = "standard"
    PRODUCTION = "production"
    RESEARCH = "research"
    DATAMOVER = "datamover"
    DATAMOVER_DEBUG = "datamover_debug"
    DEBUG = "debug"


@dataclass
class RunConfig:
    dry: bool = False
    submit_script: str = "./submit/ebi/submit.sh"
    script_with_args: str = "apps/train.py"
    script_overrides: str | None = None
    memory: int = 32
    time: float = 3.0
    job_name: Optional[str] = None
    cpu_num: Optional[int] = None
    task_num: Optional[int] = None
    node_num: Optional[int] = None
    gpu: bool = True
    gpu_num: int = 1
    gpu_type: str = "a100"
    cuda_version: str = "11.8.0"
    partition: Optional[str] = None
    stdout: str = "slurm/slurm-%j.out"
    stderr: str = "slurm/slurm-%j.err"
    mail_type: Optional[str] = None


def parse_time(time_in_hours: float) -> str:
    
    hour = math.floor(time_in_hours)
    minutes = math.floor((time_in_hours - hour) * 60)
    
    return f"--time={hour:02d}:{minutes:02d}:00"


def parse_memory(memory_in_gb: int) -> str:
    
    return f"--mem={memory_in_gb}G"


def parse_gpu(gpu_num: int, gpu_type: str) -> str:
    
    gpu_type = GPUType(gpu_type).value
    
    return f"--gres=gpu:" + gpu_type + ":" + str(gpu_num)


def build_header(run_cfg: RunConfig) -> str:
    
    sbatch = "#SBATCH "
    header = ""
    header += sbatch + parse_time(run_cfg.time) + "\n"
    header += sbatch + parse_memory(run_cfg.memory) + "\n"
    if run_cfg.gpu:
        header += sbatch + parse_gpu(run_cfg.gpu_num, run_cfg.gpu_type) + "\n"
    
    if run_cfg.job_name is not None:
        job_name_flag = f"-J={run_cfg.job_name}"
        header += sbatch + job_name_flag + "\n"
    
    if run_cfg.cpu_num is not None:
        cpu_flag = f"-c={int(run_cfg.cpu_num)}"
        header += sbatch + cpu_flag + "\n"
    
    if run_cfg.task_num is not None:
        task_flag = f"-n={int(run_cfg.task_num)}"
        header += sbatch + task_flag + "\n"
    
    if run_cfg.node_num is not None:
        node_flag = f"-N={int(run_cfg.node_num)}"
        header += sbatch + node_flag + "\n"
    
    stderr_flag = f"--error={run_cfg.stderr}"
    header += sbatch + stderr_flag + "\n"
    stdout_flag = f"--output={run_cfg.stdout}"
    header += sbatch + stdout_flag + "\n"
    if run_cfg.mail_type is not None:
        mailtype_flag = f"--mail-type={MailType(run_cfg.mail_type).value}"
        header += sbatch + mailtype_flag + "\n"
    
    if run_cfg.partition is not None:
        partition_flag = f"--partition={Partition(run_cfg.partition).value}"
        header += sbatch + partition_flag + "\n"
    
    return header


def main():
    
    default_cfg = OmegaConf.structured(RunConfig)
    cli_cfg = OmegaConf.from_cli()
    
    if hasattr(cli_cfg, "config"):
        file_cfg = OmegaConf.load(cli_cfg.config)
        del cli_cfg.config
    else:
        file_cfg = default_cfg

    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg) # type: ignore
    cfg: RunConfig = OmegaConf.to_object(cfg) # type: ignore
    
    header = build_header(cfg)

    python_command = f"python {cfg.script_with_args}"
    if cfg.gpu and (cfg.gpu_num > 1):
        python_command = f"torchrun --standalone --nproc-per-node={cfg.gpu_num} {cfg.script_with_args}"
    
    script_lines = [
        "#!/bin/bash\n",
        header + "\n",
        "source ~/.bashrc\n",
        "set -a\n",
        "source .env\n",
        "set +a\n",
        f"module load cuda/{cfg.cuda_version}\n",
        f"echo 'activate env:'\n",
        f"echo $PYTHON_ENV\n",
        f"micromamba activate $PYTHON_ENV\n",
        python_command + " $@\n"  # $1 passes first command-line argument
    ]

    tmp_dir = "/dev/shm" if os.path.exists("/dev/shm") else "."
    fd, tmp_path = tempfile.mkstemp(
        prefix="sbatch_",
        suffix=".sh",
        dir=tmp_dir,
        text=True
    )
    with os.fdopen(fd, 'w') as f:
        f.writelines(script_lines)
    print(f"submit script written to {tmp_path}")
    os.chmod(tmp_path, 0o755)   
    sbatch_command = f"sbatch {tmp_path}"
    print_command = f"[gpu:{cfg.gpu};gpu_type:{cfg.gpu_type};gpu_num:{cfg.gpu_num};time:{cfg.time};memory:{cfg.memory}]"
    print_command += " " + python_command 
    
    overrides = []
    if cfg.script_overrides is not None:
        print(f"loading overrides from {cfg.script_overrides}")
        with open(cfg.script_overrides, "r") as f:
            overrides = yaml.safe_load(f)

    if len(overrides) > 0:

        if isinstance(overrides, list):
            parsed_overrides = overrides
        elif isinstance(overrides, dict):
            parsed_overrides = list()
            for k, vs in overrides.items():
                parsed_overrides.append([f"{k}={v}" for v in vs])
            parsed_overrides = list(itertools.product(*overrides))
            parsed_overrides = [" ".join(override) for override in parsed_overrides]
        else:
            raise ValueError
        
        for override in parsed_overrides:
            try:
                print(print_command + f" '{override}'")
                if not cfg.dry:
                    subprocess.run(sbatch_command + f" '{override}'", check=True, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"Script exited with error: {e.returncode}")
                sys.exit(e.returncode)
    else:
        try:
            print(print_command)
            if not cfg.dry:
                subprocess.run(sbatch_command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Script exited with error: {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
