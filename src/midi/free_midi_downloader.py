import os
from tqdm import *

import requests
from bs4 import BeautifulSoup
from multiprocessing.dummy import Pool

START_ID = 59
END_ID = 27701

URL_PATH = "https://freemidi.org/getter-{}"
FILENAME_FOLDER = os.path.join("data","dataset","Free midi dataset")

def download_midi(id):

    url = URL_PATH.format(id)
    filename = os.path.join(FILENAME_FOLDER, "{}.mid".format(id-START_ID+1))
    user_agent = {'User-agent': 'Mozilla/5.0'}

    try:
        s = requests.Session()

        r = s.get(url, timeout=1)

        soup = BeautifulSoup(r.content, 'html.parser')
        for a in soup.select('a'):
            if a.text == 'Download MIDI':
                href = "https://freemidi.org/" + a['href']
                r = s.get(href, headers=user_agent, timeout=1)
                with open(filename, "wb") as handle:
                    handle.write(r.content)

    except:
        pass

if __name__ == '__main__':

    os.chdir('..')
    os.chdir('..')

    if not os.path.exists(FILENAME_FOLDER):
        os.makedirs(FILENAME_FOLDER)

    print("Download started")

    with Pool(processes=100) as p:
        count = END_ID - START_ID + 1
        with tqdm(total=count) as pbar:
            for _ in p.imap_unordered(download_midi, range(START_ID, END_ID+1, 1)):
                pbar.update()

    print("Download ended")
