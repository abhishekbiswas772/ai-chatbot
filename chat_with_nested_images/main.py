from img_loader import ImgHandler
from qna import QNAHandler


def main1():
    ImgHandler.load_base_image_and_embedding()


def main2(file_path):
   return QNAHandler.make_qna(file_path)


if __name__ == "__main__":
    pass
    # main1()
    main2()