from os import path

class CacheBase(object):
    def __init__(self, cache_file_path):
        self.values = []
        self.cache_file_path = cache_file_path
        self.__restore_cache(self.cache_file_path)

    def __restore_cache(self, cache_file_path):
        if not path.isfile(cache_file_path):
            return
        with open(cache_file_path, 'r') as cache_file:
            for line in cache_file:
                line = line.strip()
                if line:
                    self.values.append(self._value_from_string(line))

    def _value_from_string(self, string):
        raise NotImplementedError

    def save_result(self, value):
        self.values.append(value)
        self.__append_to_file(value)

    def read(self, *args):
        raise NotImplementedError

    def __append_to_file(self, value):
        with open(self.cache_file_path, 'a') as cache_file:
            cache_file.write(value.to_string() + '\n')
