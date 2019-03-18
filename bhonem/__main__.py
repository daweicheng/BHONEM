from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


import logging

logger = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument("-m", "--method",  required=True,
                        choices=['other'],
                        help='The processing method: other')
    parser.add_argument("-train", "--train", default=None,
                        help="The train data")
    parser.add_argument("-test", "--test", default=None,
                        help="The test data")
    args = parser.parse_args()
    return args


def main(args):
    # todo add function call here
    pass


if __name__ == "__main__":
    main(parse_args())
