from transformers import YolosConfig


def run():
    yolos_config = YolosConfig.from_pretrained("hustvl/yolos-tiny")
    print(yolos_config)


if __name__ == "__main__":
    run()
