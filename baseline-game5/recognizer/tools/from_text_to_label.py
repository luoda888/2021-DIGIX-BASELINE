# -*- coding=utf-8 -*-
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_train_file_path', type=str,
                        default='/path/to/tmp_data/recognizer_txts/train.txt')
    parser.add_argument('--dst_train_file_path', type=str,
                        default='/path/to/tmp_data/recognizer_txts/real_train.txt')

    parser.add_argument('--dictionary_file_path', type=str,
                        default='recognizer/tools/dictionary/chars.txt')

    opt = parser.parse_args()
    src_train_file_path = opt.src_train_file_path
    dst_train_file_path = opt.dst_train_file_path
    dictionary_file_path = opt.dictionary_file_path
    char_to_index = dict()
    with open(dictionary_file_path, 'r', encoding='utf-8') as in_file:
        lines = in_file.readlines()
        for index, line in enumerate(lines):
            line = line.strip('\r').strip('\n')
            char_to_index[line] = index

    with open(dst_train_file_path, 'a') as out_file:
        with open(src_train_file_path, 'r', encoding='utf-8') as in_file:
            lines = in_file.readlines()
            for index, line in enumerate(lines):
                line = line.strip('\r').strip('\n')
                line_list = line.split('\t')
                if '#' in line_list[1]:
                    continue
                if line_list[0].split('.')[1] != 'jpg':
                    print(index, line)
                if len(line_list[-1]) <= 0:
                    continue
                out_file.write('{}\t'.format(line_list[0]))
                for char in line_list[-1][:len(line_list[-1]) - 1]:
                    out_file.write('{} '.format(char_to_index[char]))
                out_file.write('{}\n'.format(char_to_index[line_list[-1][-1]]))


'''bash
python recognizer/tools/from_text_to_label.py
'''