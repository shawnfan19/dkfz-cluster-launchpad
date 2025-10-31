import os
from enum import Enum
import subprocess
import sys
from dataclasses import dataclass, field
import itertools
from omegaconf import OmegaConf
import yaml
import tempfile


class CPUQueue(Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"
    VERY_LONG = "very-long"
    HIGH_MEM = "high-mem"
    INTERACTIVE = "interactive"
    INTERACTIVE_LONG = "interactive-long"


class GPUQueue(Enum):
    DEBIAN = "gpu-debian"
    LOWPRIO_DEBIAN = "gpu-lowprio-debian"
    INTERACTIVE = "gpu-default-interactive"


@dataclass
class RunConfig:
    dry: bool = False
    script_with_args: str = "apps/train.py"
    script_overrides: None | str = None
    memory: int = 16
    gpu: bool = True
    gpu_num: int = 1
    j_exclusive: bool = True
    gpu_mem: int = 10
    cuda_version: str = "11.7"
    queue: str = "gpu-debian"
    stdout: None | str = None
    stderr: None | str = None
    blacklist: list = field(default_factory=list)
    whitelist: list = field(default_factory=list)



def validate_run_config(run_config: RunConfig):
    
    if run_config.gpu:
        assert "gpu" in run_config.queue, \
            "must submit GPU queue if GPU is requested"
    
    pass


def parse_memory_config(run_cfg: RunConfig) -> str:
    return f"rusage[mem={run_cfg.memory}GB]"


def parse_blacklist_config(run_cfg: RunConfig) -> str:
    blacklist = ""
    for node in run_cfg.blacklist:
        blacklist += f"hname!='{node}'"
        blacklist += " && "
    return blacklist[:-4]


def parse_whitelist_config(run_cfg: RunConfig) -> str:
    whitelist = ""
    for node in run_cfg.whitelist:
        whitelist += f"hname='{node}'"
        whitelist += " || "
    return whitelist[:-4]


def parse_queue_config(run_cfg: RunConfig) -> str:
    
    if run_cfg.gpu:
        return GPUQueue(run_cfg.queue).value
    else:
        return CPUQueue(run_cfg.queue).value


def parse_gpu_config(run_cfg: RunConfig) -> str:

    if run_cfg.gpu:
        num = f"num={run_cfg.gpu_num}:"
        j_exclusive = "j_exclusive=yes:" if run_cfg.j_exclusive else ""
        gmem = f"gmem={run_cfg.gpu_mem}G"
        return f"{num}{j_exclusive}{gmem}"
    else:
        return ""


def main():

    cli_cfg = OmegaConf.from_cli()
    default_cfg = OmegaConf.structured(RunConfig)
    if hasattr(cli_cfg, "config"):
        file_cfg = OmegaConf.load(cli_cfg.config)
        del cli_cfg.config
    else:
        file_cfg = default_cfg

    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg) # type: ignore
    cfg: RunConfig = OmegaConf.to_object(cfg) # type: ignore

    validate_run_config(cfg)

    bsub_command = "bsub"
    
    bsub_command += f" -R "
    bsub_command += "\""
    
    memory_args = parse_memory_config(cfg)
    bsub_command += memory_args
        
    blacklist = parse_blacklist_config(cfg)
    if blacklist != "":
        bsub_command += f" select[{blacklist}]"
    
    whitelist = parse_whitelist_config(cfg)
    if whitelist != "":
        bsub_command += f" select[{whitelist}]"
    
    bsub_command += "\""
    
    if cfg.gpu:
        gpu_args = parse_gpu_config(cfg)
        bsub_command += f" -gpu {gpu_args}"

    if cfg.stdout:
        os.makedirs(os.path.dirname(cfg.stdout), exist_ok=True)
        bsub_command += f" -o {cfg.stdout}"

    if cfg.stderr:
        os.makedirs(os.path.dirname(cfg.stderr), exist_ok=True)
        bsub_command += f" -e {cfg.stderr}"

    queue = parse_queue_config(cfg)
    bsub_command += f" -q {queue}"

    python_command = f"python {cfg.script_with_args}"
    if cfg.gpu and (cfg.gpu_num > 1):
        python_command = f"torchrun --standalone --nproc-per-node={cfg.gpu_num} {cfg.script_with_args}"
    
    script_lines = [
        "#!/bin/bash\n",
        "source ~/.bashrc\n",
        "set -a\n",
        "source .env\n",
        "set +a\n",
        f"export CUDA_HOME=/usr/local/cuda-{cfg.cuda_version}\n",
        f"export CUDA_CACHE_DISABLE=1\n",
        f"echo 'activate env:'\n",
        f"echo $PYTHON_ENV\n",
        f"micromamba activate $PYTHON_ENV\n",
        python_command + " $@\n",
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

    bsub_command += (
        f' /bin/bash -l -c "{tmp_path}'
    )

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
            parsed_overrides = list(itertools.product(*parsed_overrides))
            parsed_overrides = [" ".join(override) for override in parsed_overrides]
        else:
            raise ValueError
        
        for override in parsed_overrides:
            command = bsub_command + " " + override + '"'
            try:
                print(command)
                if not cfg.dry:
                    subprocess.run(command, check=True, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"Script exited with error: {e.returncode}")
                sys.exit(e.returncode)
    else:
        command = bsub_command + '"'
        try:
            print(command)
            if not cfg.dry:
                subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Script exited with error: {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
