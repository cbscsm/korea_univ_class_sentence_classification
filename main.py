import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import argparse
import json
import collections
from datetime import datetime

import logging

from train import char_lstm_classifier, cnn_classifier

def init_logger(path:str):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.DEBUG)
    debug_fh = logging.FileHandler(os.path.join(path, "debug.log"))
    debug_fh.setLevel(logging.DEBUG)

    info_fh = logging.FileHandler(os.path.join(path, "info.log"))
    info_fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    info_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s')
    debug_formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s | %(lineno)d:%(funcName)s')

    ch.setFormatter(info_formatter)
    info_fh.setFormatter(info_formatter)
    debug_fh.setFormatter(debug_formatter)

    logger.addHandler(ch)
    logger.addHandler(debug_fh)
    logger.addHandler(info_fh)

    return logger

def train_model(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams_dict["root_dir"], "%s/" % timestamp)

    logger = init_logger(root_dir)
    logger.info("Loaded hyper-parameter configuration from file: %s" %hparams_path)
    logger.info("Hyper-parameters: %s" %str(hparams_dict))
    hparams_dict["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    with open(os.path.join(root_dir, "hparams.json"), "w") as f_handle:
        json.dump(hparams._asdict(), f_handle, indent=2)

    # Build graph
    model = builder_class(hparams, args.data)
    model.train()


def load_and_evaluate(args, builder_class):
    hparams_path = args.hparams

    with open(hparams_path, "r") as f_handle:
        hparams_dict = json.load(f_handle)

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    root_dir = os.path.join(hparams_dict["root_dir"], "%s/" % timestamp)

    logger = init_logger(root_dir)
    logger.info("Loaded hyper-parameter configuration from file: %s" % hparams_path)
    logger.info("Hyper-parameters: %s" %str(hparams_dict))
    hparams_dict["root_dir"] = root_dir

    hparams = collections.namedtuple("HParams", sorted(hparams_dict.keys()))(**hparams_dict)

    with open(os.path.join(root_dir, "hparams.json"), "w") as f_handle:
        json.dump(hparams._asdict(), f_handle, indent=2)

    # Build graph
    model = builder_class(hparams, args.data)
    model.evaluate(args.predict)
    # model.evaluate(args.predict)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Named Entity Tagger.")
    arg_parser.add_argument("--hparams", dest="hparams", required=True,
                            help="Path to the file that contains hyper-parameter settings. (JSON format)")
    arg_parser.add_argument("--data", dest="data", type=str, required=True,
                            help="Directory that contains dataset files.")
    arg_parser.add_argument("--evaluate", dest="evaluate", type=str, default=None,
                            help="Path to the saved model.")
    args = arg_parser.parse_args()


    if args.evaluate is not None:
        load_and_evaluate(args, char_lstm_classifier.TextClassifier)
    else:
        train_model(args, char_lstm_classifier.TextClassifier)