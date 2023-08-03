import sys
import argparse
import timeit
import shutil


def main(args):
    print(">>> [COPYING] STARTED")
    source_folder = args.copy_from_dir
    destination_folder = args.copy_to_dir
    start = timeit.default_timer()

    if args.copy_to_dir and args.copy_from_dir: 
        try:
            shutil.copytree(source_folder, destination_folder, dirs_exist_ok=False)
        except FileExistsError as err:
            print("[FileExistsError] with content", err)

    stop = timeit.default_timer()
    print(">>> [COPYING] FINISHED on", (stop - start))


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--copy_to_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    parser.add_argument('--copy_from_dir', type=str,
        help='Path to the data directory containing aligned LFW face patches.')
    return parser.parse_args(argv)

if __name__ == "__main__":
    main(parse_arguments(sys.argv[1:]))



