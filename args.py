import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Filter Desambiguation')

    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs (default: 1)')
    parser.add_argument('--update_batch_size', type=int, default=5,
                        help='Batch Size per step (default: 5)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Fixed manual seed (default: 42)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--meta_lr', type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--pre_iter', type=int, default=10,
                        help="pre iterations of model")
    parser.add_argument('--meta_iter', type=int, default=10,
                        help="meta iterations of model")
    parser.add_argument('--oracle', action='store_true',
                        help="Decides wheter turn on Oracle or not!")




    args = parser.parse_args()

    return args
