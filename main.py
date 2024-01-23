import argparse
import os
from logging import getLogger

import torch
import wandb
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, init_seed, set_color

from models.afd_gccf import AFD_GCCF
from models.afd_hmlet import AFD_HMLET
from models.afd_lightgcn import AFD_LightGCN
from models.afd_ngcf import AFD_NGCF
from models.gccf import GCCF
from models.hmlet import HMLET
from models.lightgcn import LightGCN
from models.ngcf import NGCF
from trainer import MyTrainer


def get_model(args):
    model_dict = {
        'lightgcn': LightGCN,
        'afd-lightgcn': AFD_LightGCN,
        'ngcf': NGCF,
        'afd-ngcf': AFD_NGCF,
        'gccf': GCCF,
        'afd-gccf': AFD_GCCF,
        'hmlet': HMLET,
        'afd-hmlet': AFD_HMLET,
    }

    model = model_dict[args.model]
    return model


def run_single_model(args):
    # get model
    model = get_model(args)

    # configurations initialization
    config = Config(
        model=model,
        dataset=args.dataset,
        config_file_list=args.config_file_list
    )
    init_seed(config['seed'], config['reproducibility'])
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_id'])

    # logger initialization
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    # dataset filtering
    dataset = create_dataset(config)
    logger.info(dataset)

    # dataset splitting
    train_data, valid_data, test_data = data_preparation(config, dataset)

    # model loading and initialization
    model = model(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # trainer loading and initialization
    trainer = MyTrainer(config, model)

    # model training
    best_valid_score, best_valid_result = trainer.fit(
        train_data, valid_data, saved=True, show_progress=config['show_progress']
    )

    # model evaluation
    test_result = trainer.evaluate(test_data, load_best_model=True, show_progress=config['show_progress'])

    logger.info(set_color('best valid ', 'yellow') + f': {best_valid_result}')
    logger.info(set_color('test result', 'yellow') + f': {test_result}')

    wandb.finish()
    return test_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='yelp-2018',
                        help='The datasets can be: ml-1m, amazon-books, gowalla-merged, yelp-2018')
    parser.add_argument('--config', type=str, default='', help='External config file name.')
    parser.add_argument('--model', type=str, default='afd-lightgcn', help='The models')

    args, _ = parser.parse_known_args()

    # Config files
    args.config_file_list = [
        'properties/overall.yaml',
    ]

    args.config_file_list.append(f'properties/{args.dataset}.yaml')

    model_config_name = args.model.split('-')[-1]
    args.config_file_list.append(f'properties/model/{model_config_name}.yaml')
    if args.config != '':
        args.config_file_list.append(args.config)

    torch.set_num_threads(4)
    run_single_model(args)
