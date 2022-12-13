import os
import requests
from PIL import Image
import torchvision.transforms as T
from cairosvg import svg2png
import pandas as pd


def download_images():
    df = pd.read_csv('Countries_code_flags.csv', sep=',', encoding="ISO-8859-1")
    for i, row in df.iterrows():
        link = row['Flag_image_url']
        if type(link) != str:
            continue
        svg_code = requests.get(link).content
        if svg_code[0] == 70:
            continue
        svg2png(bytestring=svg_code, write_to=f'flags/rgb/img{i}.png')



if __name__ == '__main__':
    # img_data = requests.get('https://upload.wikimedia.org/wikipedia/commons/3/36/Flag_of_Albania.svg').content
    # with open('image_name.svg', 'wb') as handler:
    #     handler.write(img_data)
    #
    # svg_code = requests.get('https://upload.wikimedia.org/wikipedia/commons/3/36/Flag_of_Albania.svg').content
    # svg2png(bytestring=svg_code, write_to='output.png')
    #
    # img = T.ToTensor()(Image.open('output.png'))
    # img = T.Resize((256, 256))(img)
    # print(img.shape)
    # T.ToPILImage()(img).show()
    # main()
    download_images()
