
deactivate
Remove-Item -Path .venv -Recurse -Force
python -m venv .venv
.venv\Scripts\activate


# Sau khi đảm bảo đúng virtual environment, cài lại OpenCV
pip install --upgrade pip
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78 --no-cache-dir
pip install torch torchvision torchaudio
pip install albumentations Pillow matplotlib scipy tqdm
pip install rich
pip install torchinfo
# Kiểm tra lại
python -c "import cv2; print(cv2.__version__)"

# Collecting images 
1. Update classes in `src/utils/collect_images.py`
2. Run the script `python src/utils/collect_images.py`

# Labelling them 
pip install label-studio 


$env:LOCAL_FILES_SERVING_ENABLED="true"
label-studio start

# Training 🦾
1. Create a checkpoints folder `mkdir checkpoints`
2. Run the training pipeline `python src/train.py`

# Running  🚀 
1. To test on your test set, update the checkpoint parameter in `test.py` then run `python src/test.py`
2. To run in real time, update the checkpoint parameter in `realtime.py` then run `python src/realtime.py`</br> 
<strong>N.B.</strong> you might need need to update your camera parameter in cv2.VideoCapture() to get the right webcam for your machine. 


# fix lỗi camera không hiện
pip uninstall -y numpy opencv-python opencv-contrib-python
pip install numpy==1.26.4
pip install opencv-python==4.8.1.78
python -c "import cv2, numpy; print(cv2.__version__, numpy.__version__)"

