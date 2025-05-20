import os
from enum import Enum
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional
import itertools
from omegaconf import OmegaConf


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
    submit_script: str = "./submit/dkfz/submit.sh"
    script: str = "apps/train.py"
    script_args: str = ""
    script_overrides: dict = field(default_factory=dict)
    memory: int = 16
    gpu: bool = True
    gpu_num: int = 1
    j_exclusive: bool = True
    gpu_mem: int = 10
    queue: str = "gpu-debian"
    stdout: Optional[str] = None
    stderr: Optional[str] = None
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
    file_cfg = OmegaConf.load(cli_cfg.config)
    # We remove 'config' attribute from config as the underlying DataClass does not have it
    del cli_cfg.config

    default_cfg = OmegaConf.structured(RunConfig)
    cfg = OmegaConf.merge(default_cfg, file_cfg, cli_cfg) # type: ignore
    cfg: RunConfig = OmegaConf.to_object(cfg) # type: ignore
    submit_script = cfg.submit_script

    validate_run_config(cfg)

    base_command = "bsub"
    
    base_command += f" -R "
    base_command += "\""
    
    memory_args = parse_memory_config(cfg)
    base_command += memory_args
        
    blacklist = parse_blacklist_config(cfg)
    if blacklist != "":
        base_command += f" select[{blacklist}]"
    
    whitelist = parse_whitelist_config(cfg)
    if whitelist != "":
        base_command += f" select[{whitelist}]"
    
    base_command += "\""
    
    if cfg.gpu:
        gpu_args = parse_gpu_config(cfg)
        base_command += f" -gpu {gpu_args}"

    if cfg.stdout:
        os.makedirs(os.path.dirname(cfg.stdout), exist_ok=True)
        base_command += f" -o {cfg.stdout}"

    if cfg.stderr:
        os.makedirs(os.path.dirname(cfg.stderr), exist_ok=True)
        base_command += f" -e {cfg.stderr}"

    queue = parse_queue_config(cfg)
    base_command += f" -q {queue}"

    base_command += (
        f' /bin/bash -l -c "{submit_script} {cfg.script} {cfg.script_args}'
    )
    print(f'{base_command}"')

    if len(cfg.script_overrides) > 0:

        arg_names = cfg.script_overrides.keys()
        arg_vals = cfg.script_overrides.values()
        arg_combos = list(itertools.product(*arg_vals))
        
        for arg_combo in arg_combos:
            override = " ".join([f"{arg_name}={arg_val}" for arg_name, arg_val in zip(arg_names, arg_combo)])
            command = base_command + " " + override + '"'
            try:
                print(command)
                if not cfg.dry:
                    subprocess.run(command, check=True, shell=True)
            except subprocess.CalledProcessError as e:
                print(f"Script exited with error: {e.returncode}")
                sys.exit(e.returncode)
    else:
        command = base_command + '"'
        try:
            print(command)
            if not cfg.dry:
                subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError as e:
            print(f"Script exited with error: {e.returncode}")
            sys.exit(e.returncode)


if __name__ == "__main__":
    main()
