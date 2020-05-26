ENVNAME="ephys_venv"
python3 -m venv $ENVNAME
source $ENVNAME/bin/activate
pip install --upgrade pip  # be sure pip is up to date in the new env.
pip install wheel  # seems to be missing (note singular)
pip install cython
# # if requirements.txt is not present, create:
# # pip install pipreqs
# # pipreqs
#
# #Then:
#
pip install -r requirements_local.txt
source $ENVNAME/bin/activate

python setup.py develop
