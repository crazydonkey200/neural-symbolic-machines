source activate tensorflow_p27
pip install -r requirements.txt
mkdir ~/projects/data/
cd ~/projects/data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=104DnlcOHc5r60-PkQOyc_mcvLSs630Sl' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=104DnlcOHc5r60-PkQOyc_mcvLSs630Sl" -O wikitable.zip && rm -rf /tmp/cookies.txt
unzip wikitable.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1lwLH4-5FRZzM9JVicy3TH6Al11bRalyg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1lwLH4-5FRZzM9JVicy3TH6Al11bRalyg" -O wikisql.zip && rm -rf /tmp/cookies.txt
unzip wikisql.zip
cd ~/projects/neural-symbolic-machines/
python setup.py develop
