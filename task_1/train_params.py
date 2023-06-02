import argparse


def get_args_parser():
    parser = argparse.ArgumentParser(description='Training parameters')

    parser.add_argument('--model', type=str, default='google/long-t5-local-base',
                        help='The name or path of the model to be trained.')
    parser.add_argument('--data_path', type=str, default='datasets/touche23_single_shot_prompt',
                        help='The path to the dataset.')
    parser.add_argument('--run_name', type=str, default='Test_T5',
                        help='The name of the run.')
    parser.add_argument('--checkpoint_save_path', type=str, default='~/models/',
                        help='The path to the directory where checkpoints will be saved.')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='The learning rate used for training.')
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='The batch size used for training.')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='The batch size used for validation and testing.')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='The maximum number of epochs to train for.')
    parser.add_argument('--log_every_n_steps', type=int, default=25,
                        help='The number of training steps to log after.')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                        help='The proportion of training steps between each validation run.')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Proportion of validation set to use.')
    parser.add_argument('--num_classes', type=int, default=20,
                        help='The number of output classes.')
    parser.add_argument('--wandb_entity', type=str, default='language-technology-project',
                        help='WandB entity to log to.')
    parser.add_argument('--force_cpu', type=int, default=1,
                        help='Forcing to use cpu when training when set to 1.')
    parser.add_argument('--prompt_mode', type=str, default='few_shot',
                        help='Prompt mode used when finetuning.')
    parser.add_argument('--neo_mode', type=int, default=0,
                        help='Whether to go absolutely overkill and use GPTNeoX')
    parser.add_argument('--longT5_mode', type=int, default=1,
                        help='Whether to use longT5')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers to use for data loading')

    args = parser.parse_args()
    return args
