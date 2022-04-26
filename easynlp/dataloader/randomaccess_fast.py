import torch
from tqdm import tqdm

class RandomAccessReader(object):

    def __init__(self, filepath, endline_character='\n', print_interval=1000):
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
        with open(self._filepath, encoding='utf-8') as f:
            while has_more:
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

    def reset_filepath(self, new_path):
        print(f'[!] make sure the raw text file are the same, only the path are changed')
        self._filepath = new_path

    def init_file_handler(self):
        self.file_handler = open(self._filepath, encoding='utf-8')

    def get_line(self, line_number):
        line_data = self._lines[line_number]
        self.file_handler.seek(line_data['position'])
        string = self.file_handler.read(line_data['length'])
        return string

if __name__ == "__main__":
    # reader = RandomAccessReader('train.txt')
    # reader.init()
    reader = torch.load('test.rar')
    print(f'[!] reader lines: {reader.size}')
    # torch.save(reader, 'test.rar')

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
