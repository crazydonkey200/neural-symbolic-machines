source activate tensorflow_p27
pip install -r requirements.txt
mkdir ~/projects/data/
cd ~/projects/data/
cp ~/projects/neural-symbolic-machines/table/wtq/wikitable.zip ~/projects/data/wikitable.zip
unzip wikitable.zip
cp ~/projects/neural-symbolic-machines/table/wikisql/wikisql.zip ~/projects/data/wikisql.zip
unzip wikisql.zip
cd ~/projects/neural-symbolic-machines/
python setup.py develop
