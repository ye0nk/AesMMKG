"""Command-line arguments with the original experiment defaults preserved."""

import argparse
from datetime import datetime


def parse_kgat_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument("--seed", type=int, default=2019, help="Random seed.")

    parser.add_argument(
        "--data_name",
        nargs="?",
        default="datasets/book",
        help="Choose a dataset from {movie, book}",
    )
    parser.add_argument("--data_dir", nargs="?", default="datasets/", help="Input data path.")

    parser.add_argument(
        "--use_pretrain",
        type=int,
        default=2,
        help="0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.",
    )
    parser.add_argument(
        "--pretrain_embedding_dir",
        nargs="?",
        default="datasets/pretrain/",
        help="Path of learned embeddings.",
    )
    parser.add_argument(
        "--pretrain_model_path",
        nargs="?",
        default="trained_model/book/model_epoch680.pth",
        help="Path of stored model.",
    )

    parser.add_argument("--cf_batch_size", type=int, default=2048, help="CF batch size.")
    parser.add_argument("--kg_batch_size", type=int, default=2048, help="KG batch size.")
    parser.add_argument(
        "--test_batch_size",
        type=int,
        default=10000,
        help="Test batch size (the user number to test every batch).",
    )

    parser.add_argument("--embed_dim", type=int, default=64, help="User / entity Embedding size.")
    parser.add_argument("--relation_dim", type=int, default=64, help="Relation Embedding size.")

    parser.add_argument(
        "--laplacian_type",
        type=str,
        default="random-walk",
        help="Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.",
    )
    parser.add_argument(
        "--aggregation_type",
        type=str,
        default="bi-interaction",
        help="Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.",
    )
    parser.add_argument(
        "--conv_dim_list",
        nargs="?",
        default="[64, 32, 16]",
        help="Output sizes of every aggregation layer.",
    )
    parser.add_argument(
        "--mess_dropout",
        nargs="?",
        default="[0.1, 0.1, 0.1]",
        help="Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.",
    )

    parser.add_argument(
        "--image_emb_path", type=str, default="./datasets/book/image_embeddings.txt"
    )
    # Use the forward hasImage relation ID after the loader adds its +2 offset.
    parser.add_argument("--has_image_rel_id", type=int, default=22)
    parser.add_argument("--image_align_lambda", type=float, default=1e-4)

    parser.add_argument(
        "--kg_l2loss_lambda", type=float, default=1e-5, help="Lambda when calculating KG l2 loss."
    )
    parser.add_argument(
        "--cf_l2loss_lambda", type=float, default=1e-5, help="Lambda when calculating CF l2 loss."
    )

    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate.")
    parser.add_argument("--n_epoch", type=int, default=1000, help="Number of epoch.")
    parser.add_argument(
        "--stopping_steps", type=int, default=10, help="Number of epoch for early stopping"
    )

    parser.add_argument(
        "--cf_print_every", type=int, default=1, help="Iter interval of printing CF loss."
    )
    parser.add_argument(
        "--kg_print_every", type=int, default=1, help="Iter interval of printing KG loss."
    )
    parser.add_argument(
        "--evaluate_every", type=int, default=10, help="Epoch interval of evaluating CF."
    )

    parser.add_argument(
        "--Ks", nargs="?", default="[5, 10, 20]", help="Calculate metric@K when evaluating."
    )

    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = "trained_model/{}/lr{}_epoch{}/{}/".format(
        args.data_name, args.lr, args.n_epoch, timestamp
    )
    args.save_dir = save_dir

    return args
