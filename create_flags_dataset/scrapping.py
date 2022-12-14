import requests
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
        svg2png(bytestring=svg_code, write_to=f'../data/flags/rgb/img{i}.png')


if __name__ == '__main__':
    download_images()
