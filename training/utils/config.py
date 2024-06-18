import json
import os.path


class Config:
    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"The expected path \"{path}\" does not seem to be valid.")

        with open(path, "r") as file:
            self.config: dict = json.load(file)

    def get(self, key: str, data: dict = None, default: any = None):
        """
        Retrieve the value of a specific first-level key from the configuration dictionary.

        :param default: The default value if nothing is found
        :param key: the key to search
        :param data: The data to search the key in. If no data is passed the internal
        self.config will be used to search.
        :return: The value of the key
        """
        local_config: dict = self.config

        if data is not None:
            local_config = data

        if key not in local_config:
            if default is not None:
                return default

            return None

        return local_config[key]

    def get_nested(self, *keys: str, default: any = None):
        """
        Retrieve the value of a any-level key from the configuration dictionary.

        :param default: The default value if nothing is found
        :param keys: Breadcrumb of all the level keys to go through, the reach the most right key. e.g. 'level1',
        'level2', 'level3'.
        :return:The value of the nested key.
        """
        if len(keys) <= 0:
            raise KeyError(f"No keys to retrieve from the config.")

        local_config: dict = self.config

        for key in keys:
            local_config = self.get(key, local_config, default=default)

        return local_config
