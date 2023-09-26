# Copyright (c) OpenMMLab. All rights reserved.
import git
from mmcv.utils import collect_env as collect_basic_env
from mmcv.utils import get_git_hash

import pyskl


def collect_env():
    env_info = collect_basic_env()
    path = "/data/os/kpfiles/"
    repo = git.Repo(path, search_parent_directories=True)
    sha = repo.head.object.hexsha
    # env_info['pyskl'] = (
    #     pyskl.__version__ + '+' + get_git_hash(digits=7))
    env_info["pyskl"] = pyskl.__version__ + "+" + sha[:7]
    return env_info


if __name__ == "__main__":
    for name, val in collect_env().items():
        print(f"{name}: {val}")
