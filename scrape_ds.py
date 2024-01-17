# https://drive.google.com/drive/folders/1sRuG9mRKo8T-8dUi4MBekXJNikqoJ_l3
# https://www.dropbox.com/s/8eszpl6sa521q1h/Vintage%20Drum%20Samples%2024bit.zip?dl=0&file_subpath=%2FVintage+Drum+Samples+24bit
# https://mega.nz/folder/IJlyiYoD#8ul1IaMx5f0R3c0pZPjKOg
# https://mega.nz/folder/EmYBhChT#rDPBupU4AQAyALSq2j3YuA



url = 'https://samples.kb6.de/downloads_en.php'

import urllib.request

masterurl = 'https://samples.kb6.de'
outdir = '/home/chris/data/audio_samples/ds'
import os


fp = urllib.request.urlopen(url)
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

print(mystr)


from bs4 import BeautifulSoup

soup = BeautifulSoup(mystr, 'html.parser')

links = soup.find_all('a')

packs = []

for l in links:
    attr = l.get('href')
    if attr[-4:] == '.rar':
        packs.append(attr)
        print(attr)


for p in packs:
    str = 'wget -P %s %s/%s' % (outdir,masterurl,p)
    os.system(str)    

wav_out_dir = '/home/chris/data/audio_samples/ds_extracted'

for file in os.listdir(outdir):
    if file.endswith(".rar"):
        cms = 'unrar x %s %s' % (os.path.join(outdir, file), wav_out_dir)
        print(cms)
        os.system(cms)

# wav files from this two directories are corrupted
# rm -r [KB6]_Yamaha_CS15D
# rm -r [KB6]_Yamaha_CS-40M