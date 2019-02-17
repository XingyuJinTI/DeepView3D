export STUDENT_ID=$(whoami)
source /home/${STUDENT_ID}/miniconda3/bin/activate shaperecon
/home/${STUDENT_ID}/miniconda3/bin/conda config --add channels conda-forge
/home/${STUDENT_ID}/miniconda3/bin/conda install shapely rtree pyembree numpy scipy
/home/${STUDENT_ID}/miniconda3/bin/conda install -c conda-forge scikit-image
pip install trimesh[all]
