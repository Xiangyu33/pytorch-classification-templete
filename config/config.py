import yaml
from easydict import EasyDict


def read_yaml(yaml_path="config/config.yml"):
    return EasyDict(yaml.load(open(yaml_path), yaml.FullLoader))


if __name__ == "__main__":
    read_yaml()
