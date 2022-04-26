import torch
import ipdb
import pickle
from tqdm import tqdm

class RandomAccessReader(object):

    def __init__(self, filepath, endline_character='\n', print_interval=10000):
        """
        :param filepath:  Absolute path to file
        :param endline_character: Delimiter for lines. Defaults to newline character (\n)
        """
        self._filepath = filepath
        self._endline = endline_character
        self._print_interval = print_interval

    @property
    def size(self):
        return len(self._lines)

    def init(self):
        lines, has_more, start_idx = [], True, 0
        line_counter = 0
        with open(self._filepath) as f:
            while has_more:
                current = f.read(1)
                if current == self._endline:
                    now_idx = f.tell()
                    lines.append({'position': start_idx, 'length': line_counter})
                    start_idx = now_idx
                    line_counter = 0
                elif current == '':
                    break
                else:
                    line_counter += 1
                if len(lines) % self._print_interval == 0:
                    print(f'[!] loaded {len(lines)} lines', end='\r')
        self._lines = lines

    def fast_init(self):
        '''faster init'''
        lines, start_idx = [], 0
        with open(self._filepath) as f:
            while True:
                current = f.readline()
                length = len(current)
                if not current:
                    break
                now_idx = f.tell()
                lines.append({'position': start_idx, 'length': length})
                start_idx = now_idx
                if len(lines) % self._print_interval == 0:
                    print(f'[!] loaded {len(lines)} lines', end='\r')
        self._lines = lines

    def init_file_handler(self):
        self.file_handler = open(self._filepath)

    def get_line(self, line_number):
        line_data = self._lines[line_number]
        # self.file_handler.seek(line_data['position'])
        # string = self.file_handler.read(line_data['length'])
        self.file_handler.seek(line_data[0])
        string = self.file_handler.read(line_data[1])
        return string

    def reset_filepath(self, new_path):
        print(f'[!] make sure raw text keep the same, only its path is changed!!!')
        self._filepath = new_path

    def save_to_text(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            for item in tqdm(self._lines):
                start_idx = item['position']
                length = item['length']
                f.write(f'{start_idx}\t{length}\n')
        print(f'[!] save the start position and length into {path}')
            
    def load_from_text(self, path, size=-1):
        with open(path, encoding='utf-8') as f:
            lines = []
            for line in f:
                start, length = line.strip().split('\t')
                start, length = int(start), int(length)
                lines.append((start, length))
                if len(lines) % self._print_interval == 0:
                    print(f'[!] load {len(lines)}', end='\r')
                if len(lines) == size:
                    break
        self._lines = lines
        print(f'[!] load {len(lines)} from {path}')


if __name__ == "__main__":
    with open('/apdcephfs/share_916081/johntianlan/chatbot-large-scale-dataset-final-version/train.rar', 'rb') as f:
        reader = pickle.load(f)
    exit()

    # test error
    reader.init_file_handler()
    error = 0
    for i in tqdm(range(1000000)):
        try:
            reader.get_line(i)
        except:
            error += 1
    print(f'[!] error num: {error}')
    if error == 0:
        print(f'[!] test perfectly with no errors')
