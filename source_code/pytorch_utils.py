"""
set_device
set_seed
"""
import torch
import numpy as np
import torch
import random
from datetime import datetime
import os
from collections import Counter
import json
import copy
import shutil
import socket


def set_device(args):
    device = None

    if args.no_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #endif
    assert device is not None

    args.device = device


def set_output_folder(args):
    """
    set unique output_folder based on time, so that each run there is a unique folder

    Nov25_07-06-03_Nianzus-MacBook-Pro-2.local
    # import socket
    # from datetime import datetime
    # current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S')
    # log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)

    :param args:
    :return:
    """
    parent_folder = args.output_dir
    current_time = datetime.now().strftime('%Y%b%d_%H-%M-%S')
    args.output_dir = os.path.join(parent_folder, f"{args.model_type}_{current_time}_{socket.gethostname()}_{args.output_tag}")

    # set up output directory
    # 1. the output_dir exists
    # 2. if it exists, os.listdir will print out all the files and directories under this path. If there is nothing
    #    in this directory, it will return empty list [], which is treated as False
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir):
        raise ValueError(
            "Output directory ({}) already exists and is not empty.".format(
                args.output_dir))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    # endif


def check_label_distribution(label_array):
    label_counter = Counter(label_array)
    print("================= label distribution =================")
    for label, num in sorted(label_counter.items(), key=lambda x: x[0]):
        print("{} -- {}".format(label, num))
    # endfor


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() > 0:
        torch.cuda.manual_seed_all(args.seed)
    pass


def save_checkpoint(state, checkpoint_dir):
    save_file = os.path.join(checkpoint_dir, "checkpoint.pt")
    print(f">>>>>>>>> save checkpoint to {save_file} <<<<<<<<<<<<")
    torch.save(state, save_file)


def load_checkpoint(args, resume_checkpoint_dir):
    resume_checkpoint_file_path = os.path.join(resume_checkpoint_dir, args.best_model_folder, args.checkpoint_file_name)
    print(f">>>>>>>>> load checkpoint from {resume_checkpoint_file_path} <<<<<<<<<")
    checkpoint = torch.load(resume_checkpoint_file_path)

    return checkpoint


# checkpoint = {"epoch": epoch_index,
# "relation_id": relation_id,
# "batch_index": batch_index,
# "global_step": global_step,
# "train_loss": total_train_loss,
# "state_dict": model.state_dict(),
# "optimizer": optimizer.state_dict()}

def load_model_from_checkpoint(model, device, model_file_path):
    if device.type == "cpu":
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        # model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    elif device.type == "cuda":
        checkpoint = torch.load(model_file_path)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        
        # model.load_state_dict(torch.load(model_file_path, map_location="cuda"))
    else:
        raise SystemError("model file cannot be loaded to device: {}".format(device))
    return model


def load_model(model, device, model_file_path):
    if device.type == "cpu":
        model.load_state_dict(torch.load(model_file_path, map_location="cpu"))
    elif device.type == "cuda":
        model.load_state_dict(torch.load(model_file_path, map_location="cuda"))
    else:
        raise SystemError("model file cannot be loaded to device: {}".format(device))
    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)
    pass


def write_params(args):
    args_dict_tmp = vars(args)
    args_dict = copy.deepcopy(args_dict_tmp)
    args_dict["device"] = args_dict["device"].type

    config_file = os.path.join(args.output_dir, 'params.json')
    with open(config_file, 'w') as f:
        f.write(json.dumps(args_dict) + '\n')

    param_file = os.path.join(args.output_dir, "params.txt")  # the same as config.jsonl but print in the nicer way
    with open(param_file, "w") as f:
        f.write("============ parameters ============\n")
        print("============ parameters =============")
        for k, v in args_dict.items():
            f.write("{}: {}\n".format(k, v))
            print("{}: {}".format(k, v))
        # endfor
        print("=====================================")
    # endwith


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def move_log_file_to_output_directory(output_dir, file_path):
    """
        The output file name is at the bash file.
        Redirect all the output to the log file.
        The reason do not use file handler of logging module is that, it could only redirect the message from
        user write script to file. For the logging information in API/pacakges, it will not be blocked and not
        shown in the stdout.
        Using redirect in linux console will redirect all the information printed out on the screen to the file
        not matter where it is from.
        :return:
        """
    shutil.move(file_path, output_dir)
    pass

