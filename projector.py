from argparse import ArgumentParser
import os

from sentence_transformers import SentenceTransformer
from torch.utils.tensorboard import SummaryWriter
import tensorboard as tb
import tensorflow as tf
import pandas as pd


def parse_args():
    parser = ArgumentParser("st_projector")

    parser.add_argument(
        "--data",
        action="store",
        required=True,
        help="Examples Tsv file with examples and labels",
    )

    parser.add_argument(
        "--main_column",
        action="store",
        required=True,
        help="Column containing sentences examples.",
    )

    parser.add_argument(
        "--st_model",
        action="store",
        required=True,
        help="Sentence-Transformers model name or path.",
    )

    parser.add_argument(
        "--log_dir",
        action="store",
        required=True,
        help="Tensorboard log_dir to keep embeddings data.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

    examples = pd.read_table(args.data)[args.main_column]
    model = SentenceTransformer(args.st_model)
    embeddings = model.encode(examples, convert_to_tensor=True)

    writer = SummaryWriter(args.log_dir)
    writer.add_embedding(embeddings, examples, tag=args.st_model)

    metadata_path = os.path.join(args.log_dir, "00000", args.st_model, "metadata.tsv")
    examples.to_csv(metadata_path, index=None, sep="\t")

    program = tb.program.TensorBoard()
    program.configure(argv=[None, "--logdir", args.log_dir])
    program = tb.launch()


if __name__ == "__main__":
    main()
