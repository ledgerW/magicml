import argparse
import os
import datetime
import logging

from utils import train


def parse_args():
    parser = argparse.ArgumentParser()

    # hyperparameters for your train.py script - MODIFY AS NEEDED
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--steps', type=int, default=85000)
    parser.add_argument('--batch_size_gpl', type=int, default=16)
    parser.add_argument('--batch_size_qgen', type=int, default=10)
    
    # Sagemaker REQUIRED params/directories - DO NOT CHANGE NAMES
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()



if __name__ == "__main__":
    args, _ = parse_args()
    logging.info(args)

    # Load training data
    # YOUR TRAINING DATA LOAD FUNCTION - MUST TAKE args.train WHICH POINTS TO YOUR DATA
    # args.train will be a local path if running local OR an S3 URI if using Sagemaker
    #train_x, train_y = get_input_data(args.train)

    # Load val/test data
    #val_x, val_y = get_input_data(args.test)

    # Load model for training
    # YOUR MODEL LOAD FUNCTION
    #model = build_model(args.dense_layers)

    # YOUR TRAIN FUNCTION DEFINED ABOVE
    # MODIFY AS NEEDED
    train(
        args.train,
        args.model_dir,
        args.output_dir,
        args.model_name,
        args.steps,
        args.batch_size_gpl,
        args.batch_size_qgen
    )
    