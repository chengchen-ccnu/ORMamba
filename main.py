from tools import run_net
from utils import parser
import torch , random
import numpy as np
import os


def main():
    # config
    args = parser.get_args()
    parser.setup(args)   
    if args.benchmark == 'MTL':
        if not args.usingDD:
            args.score_range = 100
    print(args)
    init_seed(args.seed)
    # run
    torch.cuda.empty_cache()
    run_net(args)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # cudnn.enabled = True
    # cudnn.benchmark = False
    # cudnn.deterministic = True

    # os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True, warn_only=True)


if __name__ == '__main__':
    main()