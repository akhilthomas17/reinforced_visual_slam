#!/usr/bin/env python3
import os
import sys
import argparse

def write_filenames_to_txt(folder, out_filename):
    folder_parent = os.path.dirname(os.path.normpath(folder))
    outfile_path = os.path.join(folder_parent, out_filename)
    print("Result will be stored in", outfile_path)
    with open(outfile_path, "w+") as train_file:
        cnt = 0
        for root, directories, filenames in os.walk(folder):
            for files in sorted(filenames):
                cnt += 1
                if cnt < 4:
                    sep = ","
                else:
                    sep = "\r\n"
                    cnt = 0
                #print(files)
                #print(os.path.basename(os.path.normpath(root)))
                train_file.write( os.path.basename(os.path.normpath(root)) + "/" + files + sep)


def run():
    parser = argparse.ArgumentParser(description=( "To walk through subdirectories of the given directory "
        "and make a single file containing relative path to each of the samples. Each line in file would be:\n"
        "depth_gt,rgb,sparseDepth,sparseDepthVar"))
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder which contains required files")
    parser.add_argument("--out_filename", type=str, required=True, help="Name of output file")
    args = parser.parse_args()
    #assert os.path.isfolder(args.folder)
    write_filenames_to_txt(args.folder, args.out_filename)


if __name__ == "__main__":
    run()
    sys.exit()