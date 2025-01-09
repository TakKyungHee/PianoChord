import subprocess
import os
import shutil
import cv2
from pdf2image import convert_from_path


# jdk-17 다운로드 코드
# import os       #importing os to set environment variable
# def install_java():
#   !apt-get install -y openjdk-17-jdk-headless -qq > /dev/null      #install openjdk
#   os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"     #set environment variable
# os.environ["PATH"] += ":/usr/lib/jvm/java-17-openjdk-amd64/bin"
#   !java -version       #check java version
# install_java()

def convert_pdf_to_images(path, input_name):
    pages = convert_from_path(input_name)
    for i, page in enumerate(pages):
        page.save(os.path.join(path, f'page{i+1}.jpg'), 'JPEG')


def image_upscaling(input_name):
    # 이미지 읽기
    image = cv2.imread(input_name)
    height, width, _ = image.shape
    if width < 2000 or height < 2000:
        # 확대할 배율 설정
        scale_factor = (2000/width+2000/height)*0.5
        # 이미지 확대 (INTER_CUBIC 또는 INTER_LANCZOS4가 더 좋은 품질을 제공)
        resized_image = cv2.resize(
            image, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LANCZOS4)
        # 결과 이미지 저장
        cv2.imwrite(input_name, resized_image)


def convert_image_to_musicxml(input_name, download_folder):
    # Audiveris JAR 파일 경로
    audiveris_jar_path = 'C:/Users/jacuz/Audiveris-5.3.1/lib/*'

    # Audiveris 명령어 구성
    command = [
        "java",
        "-cp",
        audiveris_jar_path,
        "org.audiveris.omr.Main",
        "-batch",
        "-export",
        "-output",
        download_folder,
        input_name
    ]

    # 명령어 실행
    subprocess.run(command)


def cleaning(path, destination):
    mxls = [(os.path.join(path, file), os.path.join(destination, file))
            for file in os.listdir(path) if file.endswith('.mxl')]
    omrs = [os.path.join(path, file)
            for file in os.listdir(path) if file.endswith('.omr')]
    logs = [os.path.join(path, file)
            for file in os.listdir(path) if file.endswith('.log')]
    for mxl, desti in mxls:
        shutil.move(mxl, desti)
    for omr in omrs:
        os.remove(omr)
    for log in logs:
        os.remove(log)


file_path = "./akbobada_sample"
destination_folder = "./xml"

if __name__ == '__main__':
    # 89524-91795, 80000-80298
    for number in range(80300, 80300):
        input_name = os.path.join(file_path, str(number)+'.png')
        if os.path.isfile(input_name):
            image_upscaling(input_name)
            convert_image_to_musicxml(input_name, file_path)
            cleaning(file_path, destination_folder)

# if __name__ == '__main__':
#     if len([True for file in os.listdir(file_path) if file.endswith('.png') or file.endswith('.jpg')]) == 0:
#         for file in os.listdir(file_path):
#             if file.endswith('.pdf'):
#                 input_name = os.path.join(file_path, file)
#                 convert_pdf_to_images(file_path, input_name)
#     if len([True for file in os.listdir(file_path) if file.endswith('.mxl')]) == 0:
#         for file in os.listdir(file_path):
#             if file.endswith('.png') or file.endswith('.jpg'):
#                 input_name = os.path.join(file_path, file)
#                 image_upscaling(input_name)
#                 convert_image_to_musicxml(input_name, file_path)
#                 cleaning(file_path, file_path)
