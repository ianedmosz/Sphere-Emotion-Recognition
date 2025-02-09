# **Libraries and Environment Configuration**

This document outlines the libraries and dependencies used in the `NeuroEmociones` environment for the **Sphere-Emotion-Recognition** project.

---

## **Python Version**
- **Python 3.8.10**

---

## **Conda Dependencies**

The following dependencies are managed via **Conda**:

```plaintext
argon2-cffi==20.1.0
async_generator==1.10
attrs==21.2.0
backcall==0.2.0
backports==1.0
backports.functools_lru_cache==2.0.0
blas==1.0
bleach==3.3.0
ca-certificates==2024.9.24
certifi==2024.8.30
cffi==1.14.5
colorama==0.4.4
cycler==0.10.0
decorator==5.0.9
defusedxml==0.7.1
entrypoints==0.3
freeglut==3.4.0
freetype==2.10.4
git==2.32.0
icu==58.2
importlib-metadata==3.10.0
intel-openmp==2021.2.0
ipykernel==5.3.4
ipython==7.22.0
ipython_genutils==0.2.0
jedi==0.17.0
jinja2==3.0.1
jsonschema==3.2.0
jupyter_client==6.1.12
jupyter_core==4.7.1
kiwisolver==1.3.1
libblas==3.9.0
libcblas==3.9.0
liblapack==3.9.0
libpng==1.6.37
libsodium==1.0.18
lz4-c==1.9.3
markupsafe==2.0.1
matplotlib==3.3.4
mkl==2021.2.0
mkl-service==2.3.0
mkl_fft==1.3.0
mkl_random==1.2.1
nbclient==0.5.3
nbconvert==6.1.0
nbformat==5.1.3
nest-asyncio==1.5.1
notebook==6.4.0
numpy==1.23.2
packaging==23.1
pandas==1.2.4
pandocfilters==1.4.3
parso==0.8.2
pickleshare==0.7.5
pillow==9.3.0
prometheus_client==0.11.0
prompt-toolkit==3.0.17
pycparser==2.20
pygments==2.9.0
pyparsing==2.4.7
python-dateutil==2.8.1
pytz==2021.1
pywin32==227
pywinpty==0.5.7
pyzmq==20.0.0
send2trash==1.5.0
setuptools==52.0.0
six==1.16.0
sqlite==3.35.4
tbb==2021.5.0
terminado==0.9.4
testpath==0.5.0
tk==8.6.10
tornado==6.1
traitlets==5.0.5
wcwidth==0.2.5
webencodings==0.5.1
wheel==0.36.2
zipp==3.5.0
zlib==1.2.11
zstd==1.4.9
m2w64-gcc-libgfortran==5.3.0
m2w64-gcc-libs==5.3.0
m2w64-gcc-libs-core==5.3.0
m2w64-gmp==6.1.0
m2w64-libwinpthread-git==5.0.0.4634.697f757
msys2-conda-epoch==20160418
olefile==0.46
openssl==1.1.1w
pandoc==2.19.2
qt==5.9.7
sip==4.19.13
tk==8.6.10
vc==14.2
vs2015_runtime==14.27.29016
xz==5.2.5
zeromq==4.3.3
zlib==1.2.11
zstd==1.4.9 

```

## **Pip Dependencies**

The following dependencies are installed via `pip`:

```plaintext
brainflow==5.15.0
joblib==1.4.2
pyeeg==0.4.4
pyserial==3.5
python-osc==1.8.3
scikit-learn==1.3.1
scipy==1.10.1
threadpoolctl==3.5.0

```

## **How to Use This Configuration**

1. **Recreate the environment:** Use the provided `environment.yml` file or install dependencies manually:
   ```bash
   conda env create -f NeuroEmociones.yml
2. **Activate the environment:** 
```bash
conda activate NeuroEmociones.yml
```
3. **Install pip dependencies:** 
```bash
pip install brainflow==5.15.0 joblib==1.4.2 pyeeg==0.4.4 pyserial==3.5 python-osc==1.8.3 scikit-learn==1.3.1 scipy==1.10.1 threadpoolctl==3.5.0 git+https://github.com/forrestbao/pyeeg.git

```

