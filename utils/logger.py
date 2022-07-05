import logging
import os

def get_logger(record_file):

    ALL_LOG_FORMAT = "%(message)s"

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tx_handler = logging.FileHandler(filename=record_file, mode='w+')
    tx_handler.setLevel(logging.INFO)
    tx_handler.setFormatter(logging.Formatter(ALL_LOG_FORMAT))

    logger.addHandler(tx_handler)

    return logger


def check_file(args):

    # 获取数据集名称与模型名称
    dataset = args.dataset
    model = args.model
    D = args.D
    Li = args.Li

    # 创建相对应的目录
    mdl_path = os.path.join(args.mdl_path, dataset)
    rec_path = os.path.join(args.rec_path, dataset)
    acc_path = os.path.join(args.acc_path, dataset)
    loss_path = os.path.join(args.loss_path, dataset)

    os.makedirs(mdl_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)

    # 根据数据集与模型设置相应的目录与文件标识名称
    args.mdl_path = os.path.join(mdl_path, "{}_{}d{}Li{}.mdl".format(dataset, model, D, Li))
    args.rec_path = os.path.join(rec_path, "{}_{}d{}Li{}log.txt".format(dataset, model, D, Li))
    args.acc_path = os.path.join(acc_path, "{}_{}d{}Li{}_acc.png".format(dataset, model, D, Li))
    args.trainloss_path = os.path.join(loss_path, "{}_{}d{}Li{}_trainloss.png".format(dataset, model, D, Li))
    args.textloss_path = os.path.join(loss_path, "{}_{}d{}Li{}_testloss.png".format(dataset, model, D, Li))

    print(args)


def check_mlp_file(args):

    # 获取数据集名称与模型名称
    dataset = args.dataset
    model = args.model
    block = args.num_blocks
    use_spin = 'T' if args.use_spin else 'F'

    # 创建相对应的目录
    mdl_path = os.path.join(args.mdl_path, dataset)
    rec_path = os.path.join(args.rec_path, dataset)
    acc_path = os.path.join(args.acc_path, dataset)
    loss_path = os.path.join(args.loss_path, dataset)

    os.makedirs(mdl_path, exist_ok=True)
    os.makedirs(rec_path, exist_ok=True)
    os.makedirs(acc_path, exist_ok=True)
    os.makedirs(loss_path, exist_ok=True)

    # 根据数据集与模型设置相应的目录与文件标识名称
    args.mdl_path = os.path.join(mdl_path, "{}_{}_b{}x{}.mdl".format(dataset, model, block, use_spin))
    args.rec_path = os.path.join(rec_path, "{}_{}_b{}x{}_log.txt".format(dataset, model, block, use_spin))
    args.acc_path = os.path.join(acc_path, "{}_{}_b{}x{}_acc.png".format(dataset, model, block, use_spin))
    args.trainloss_path = os.path.join(loss_path, "{}_{}_b{}x{}_trainloss.png".format(dataset, model, block, use_spin))
    args.textloss_path = os.path.join(loss_path, "{}_{}_b{}x{}_testloss.png".format(dataset, model, block, use_spin))

    print(args)