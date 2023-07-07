import os
import json
import errno
import argparse

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='CUHK-PEDES', type=str)
    parser.add_argument('--dataset_root_dir', default='./CUHK-PEDES', type=str)
    args = parser.parse_args()
    raw_annotation_file_name = ""
    if args.dataset_name == "CUHK-PEDES":
        raw_annotation_file_name = "reid_raw.json"
    elif args.dataset_name == "ICFG-PEDES":
        raw_annotation_file_name = "ICFG-PEDES.json"
    elif args.dataset_name == "RSTPReid":
        raw_annotation_file_name = "data_captions.json"
    raw_annotation_file_path = os.path.join(args.dataset_root_dir, raw_annotation_file_name)
    # split raw annotations into training, validation and test dataset
    anns = json.load(open(raw_annotation_file_path, "r"))
    train = []
    val = []
    test = []
    for ann in anns:
        if args.dataset_name == "RSTPReid":
            ann['file_path'] = ann.pop('img_path')
        eval(ann['split']).append(ann)
    output_dir = os.path.join(args.dataset_root_dir, "processed_data")
    mkdir_if_missing(output_dir)
    json.dump(train, open(os.path.join(output_dir, "train.json"), 'w'))
    json.dump(val, open(os.path.join(output_dir, "val.json"), 'w'))
    json.dump(test, open(os.path.join(output_dir, "test.json"), 'w'))
