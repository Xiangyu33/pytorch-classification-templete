from model.FaceAttriClassifier import FaceAttriClassifier


def build_model(config):
    return FaceAttriClassifier(config)


if __name__ == "__main__":
    from config import read_yaml

    cfg = read_yaml()
    build_model(cfg)
