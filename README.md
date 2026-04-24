chạy 
cài thư viện

C:/Users/Phuoc/AppData/Local/Programs/Python/Python311/python.exe -m venv .venv
.venv\Scripts\activate
Python 3.11.9
pip install -r requirements.txt

chạy file



chụp dataset: 


python src/dataset_collection/sign_dataset_tool.py capture --num-per-mode 15 --capture-interval 0.65 --hand-appear-delay 1.2 --blur-threshold 40 --dedup-hash-distance 8 --dedup-window 8



# Review + chinh sua bbox
.venv\Scripts\python.exe src/dataset_collection/sign_dataset_tool.py review --class-name hello



opencv-python==4.13.0.92
mediapipe==0.10.14
numpy==2.4.4
rich==15.0.0
