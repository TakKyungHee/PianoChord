import os
import requests
requests.packages.urllib3.disable_warnings()

# 상태 코드 확인, 91795-88498, 80000-81001
for number in range(88498, 87000, -1):
    # URL 지정
    url = "https://www.akbobada.com/musicDetail.html?musicID="+str(number)
    keyword = '피아노 3단 악보</font><br /><img  src='
    # HTTP GET 요청 보내기
    response = requests.get(url, verify=False)
    if response.status_code == 200:
        # HTML 콘텐츠 출력 (또는 다른 처리를 위해 저장)
        html_content = response.text
        download_folder = "./akbobada_sample"
        file_name = os.path.join(download_folder, str(number)+'.png')
        os.makedirs(download_folder, exist_ok=True)
        startindex = 0
        while True:
            startindex = html_content.find(
                keyword, startindex)+len(keyword)+1
            if startindex == len(keyword):
                break
            endindex = html_content.find(
                ' ', startindex)-1
            file_url = requests.get(
                html_content[startindex:endindex], verify=False)
            with open(file_name, 'wb') as file:
                file.write(file_url.content)
            print(f"File downloaded: {file_name}")
