chạy 
cài thư viện

C:/Users/Phuoc/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.venv\Scripts\activate
Python 3.11.9
pip install -r requirements.txt

chạy file



chụp dataset: 


python src/dataset_collection/sign_dataset_tool.py capture --num-per-mode 15 --capture-interval 0.65 --hand-appear-delay 1.2 --blur-threshold 80 --dedup-hash-distance 8 --dedup-window 8




.venv\Scripts\python.exe src/dataset_collection/sign_dataset_tool.py capture
.venv\Scripts\python.exe src/dataset_collection/sign_dataset_tool.py capture --blur-threshold 90
.venv\Scripts\python.exe src/dataset_collection/sign_dataset_tool.py capture --hand-appear-delay 3

# Review + chinh sua bbox
.venv\Scripts\python.exe src/dataset_collection/sign_dataset_tool.py review --class-name hello

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
.venv\Scripts\python.exe -m pip uninstall -y numpy opencv-python opencv-contrib-python mediapipe
.venv\Scripts\python.exe -m pip install --no-cache-dir numpy==1.26.4 opencv-python==4.8.1.78 opencv-contrib-python==4.8.1.78 mediapipe==0.10.14
.venv\Scripts\python.exe -c "import cv2, mediapipe, numpy; print(cv2.__version__, mediapipe.__version__, numpy.__version__)"

