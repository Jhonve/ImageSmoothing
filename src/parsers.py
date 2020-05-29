import argparse

def getParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=24, help="Size of one batch.")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate.")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of processes when load data.")
    parser.add_argument("--data_path", type=str, default="../../Datasets/HairImgs/", help="Path to dataset.")
    parser.add_argument("--data_path_file", type=str, default="./DataPath.h5", help="Path to data_path H5 file.")

    parser.add_argument("--num_epoch", type=int, default=24, help="Number of train epoch.")
    parser.add_argument("--num_val_batch", type=int, default=32, help="Number of selected batch for validation.")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/", help="Path to the checkpoints.")
    parser.add_argument("--val_path", type=str, default="./validations/", help="Path to the validation results.")
    parser.add_argument("--test_path", type=str, default="./testimgs/", help="Path to the test results.")
    parser.add_argument("--current_model", type=str, default="./checkpoints/5_model.t7", help="Latest trainded model.")
    # parser.add_argument("--current_model", type=str, default="", help="Latest trainded model.")

    opt, _ = parser.parse_known_args()

    return opt