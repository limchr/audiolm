# https://drive.google.com/drive/folders/1sRuG9mRKo8T-8dUi4MBekXJNikqoJ_l3
# https://www.dropbox.com/s/8eszpl6sa521q1h/Vintage%20Drum%20Samples%2024bit.zip?dl=0&file_subpath=%2FVintage+Drum+Samples+24bit
# https://mega.nz/folder/IJlyiYoD#8ul1IaMx5f0R3c0pZPjKOg
# https://mega.nz/folder/EmYBhChT#rDPBupU4AQAyALSq2j3YuA


import urllib.request
import os
from bs4 import BeautifulSoup



def extract_archives(archives_dir, extract_dir):
    for file in os.listdir(archives_dir):
        if file.endswith(".rar"):
            cms = 'unrar x "%s" "%s"' % (os.path.join(archives_dir, file), extract_dir)
            print(cms)
            os.system(cms)
        if file.endswith(".zip"):
            cms = 'unzip "%s" -d "%s"' % (os.path.join(archives_dir, file), extract_dir)
            print(cms)
            os.system(cms)


url = 'https://samples.kb6.de/downloads_en.php'


masterurl = 'https://samples.kb6.de'
archive_dir = '/home/chris/data/audio_samples/ds'
wav_out_dir = '/home/chris/data/audio_samples/ds_extracted'



fp = urllib.request.urlopen(url)
mybytes = fp.read()

mystr = mybytes.decode("utf8")
fp.close()

print(mystr)



soup = BeautifulSoup(mystr, 'html.parser')

links = soup.find_all('a')

packs = []

for l in links:
    attr = l.get('href')
    if attr[-4:] == '.rar':
        packs.append(attr)
        print(attr)


for p in packs:
    str = 'wget -P %s %s/%s' % (archive_dir,masterurl,p)
    os.system(str)    



extract_archives(archive_dir, wav_out_dir)




# other sample packs
input_dir = '/home/chris/data/audio_samples/SAMPLE PACKS/'

extract_archives(input_dir, wav_out_dir)



# wav files from this two directories are corrupted
# rm -r [KB6]_Yamaha_CS15D
# rm -r [KB6]_Yamaha_CS-40M