import sys
import os

print("DEBUG: Script started", flush=True)
print(f"DEBUG: Python version: {sys.version}", flush=True)
print(f"DEBUG: Platform: {sys.platform}", flush=True)

# Set unbuffered mode via environment variable
os.environ['PYTHONUNBUFFERED'] = '1'

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', message='.*MINGW.*')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("DEBUG: Importing dependencies...", flush=True)

# First check if DLLs can be loaded
try:
    import ctypes
    print("DEBUG: ctypes available", flush=True)
except Exception as e:
    print(f"WARNING: ctypes not available: {e}", flush=True)

try:
    import cv2
    print(f"DEBUG: cv2 imported successfully, version: {cv2.__version__}", flush=True)
except ImportError as e:
    print(f"FATAL: Failed to import cv2 - ImportError: {e}", flush=True)
    print("Try: pip uninstall opencv-python -y && pip install opencv-python==4.8.1.78", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"FATAL: Failed to import cv2 - {type(e).__name__}: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    import numpy as np
    print("DEBUG: numpy imported successfully", flush=True)
except Exception as e:
    print(f"FATAL: Failed to import numpy: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

import uuid
import time 

print("DEBUG: Core imports successful", flush=True)

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

print(f"DEBUG: Python path: {sys.path[:3]}", flush=True)

# Try to import custom modules with fallback
try:
    from setup import get_classes
    print("DEBUG: Successfully imported setup.get_classes", flush=True)
except ImportError as e:
    print(f"Warning: Could not import setup.get_classes(). Error: {e}", flush=True)
    print("Using default classes.", flush=True)
    def get_classes():
        return ['hello', 'thanks', 'yes', 'no', 'iloveyou']

try:
    from logger import logger
    print("DEBUG: Successfully imported logger", flush=True)
except ImportError as e:
    print(f"Warning: Could not import logger. Error: {e}", flush=True)
    print("Using basic print statements.", flush=True)
    # Create a simple logger fallback
    class SimpleLogger:
        def print_banner(self): pass
        def capture(self, msg): print(f"[CAPTURE] {msg}", flush=True)
        def capture_error(self, cls, msg): print(f"[ERROR] {cls}: {msg}", flush=True)
        def success(self, msg): print(f"[SUCCESS] {msg}", flush=True)
        def info(self, msg): print(f"[INFO] {msg}", flush=True)
        def warning(self, msg): print(f"[WARNING] {msg}", flush=True)
        def capture_session_start(self, classes, num, sleep): 
            print(f"[SESSION] Starting capture for {len(classes)} classes, {num} images each", flush=True)
        def capture_class_start(self, cls, num): 
            print(f"[CLASS] Starting capture for {cls}: {num} images", flush=True)
        def capture_success(self, cls, idx): 
            print(f"[SUCCESS] {cls} - Image {idx} captured", flush=True)
        def create_capture_progress(self, total, desc):
            from contextlib import contextmanager
            @contextmanager
            def progress():
                class DummyProgress:
                    def add_task(self, desc, total): return 0
                    def update(self, task, advance): pass
                yield DummyProgress()
            return progress()
        def capture_session_complete(self, total, num_classes):
            print(f"[COMPLETE] Captured {total} total images across {num_classes} classes", flush=True)
    logger = SimpleLogger()

classes = get_classes()
print(f"DEBUG: Classes loaded: {classes}", flush=True)
print(f"DEBUG: Number of classes: {len(classes) if classes else 0}", flush=True)

class CaptureImages(): 
    def __init__(self, path: str, classes: dict, camera_id: int) -> None: 
        self.cap = cv2.VideoCapture(camera_id) 
        self.path = path 
        self.classes = classes
        
        # Initialize logger and show banner
        logger.print_banner()
        logger.capture("Image capture system initialized")
        
        # Verify camera connection
        if not self.cap.isOpened():
            logger.capture_error("Camera", f"Could not open camera {camera_id}")
            raise Exception(f"Could not open camera {camera_id}")
        else:
            logger.success(f"Camera {camera_id} connected successfully")
        
        # Ensure output directory exists
        os.makedirs(self.path, exist_ok=True)
        logger.info(f"Output directory: {self.path}")

    def capture(self, class_name: str) -> bool:     
        try: 
            ret, frame = self.cap.read() 
            raw_frame = frame.copy()
            if not ret:
                raise Exception("Failed to read from camera")
                
            image = cv2.putText(frame, f'Capturing {class_name}', (0,100), cv2.FONT_HERSHEY_DUPLEX, 3, (0,0,0), 2, cv2.LINE_AA)
            cv2.imshow('Image Capture', image)
            
            # Generate unique filename
            filename = f'{class_name}-{uuid.uuid1()}.jpg'
            filepath = os.path.join(self.path, filename)
            cv2.imwrite(filepath, raw_frame)
            
            if cv2.waitKey(1) & 0xFF==ord('q'):
                logger.warning("Quit key pressed - stopping capture")
                return False
                
            return True
            
        except Exception as e: 
            logger.capture_error(class_name, str(e))
            return False

    def run(self, sleep_time: int = 1, num_images: int = 10):
        # Display session information
        logger.capture_session_start(self.classes, num_images, sleep_time)
        
        total_captured = 0
        
        for class_idx, img_class in enumerate(self.classes): 
            logger.capture_class_start(img_class, num_images)
            
            # Create progress bar for this class
            with logger.create_capture_progress(num_images, img_class) as progress:
                class_task = progress.add_task(f"Capturing {img_class}", total=num_images)
                
                class_captured = 0
                for idx in range(num_images): 
                    success = self.capture(img_class)
                    
                    if success:
                        class_captured += 1
                        total_captured += 1
                        logger.capture_success(img_class, idx + 1)
                        progress.update(class_task, advance=1)
                    else:
                        logger.capture_error(img_class, f"Image {idx + 1}")
                        # Still advance progress to continue
                        progress.update(class_task, advance=1)
                    
                    time.sleep(sleep_time)
                
                # Show completion for this class
                
                logger.success(f"Completed {img_class}: {class_captured}/{num_images} images captured")
                time.sleep(3)

        # Show session completion
        logger.capture_session_complete(total_captured, len(self.classes))
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released and windows closed")

if __name__ == '__main__': 
    try:
        print("=" * 60, flush=True)
        print("Starting image capture system...", flush=True)
        print(f"Classes to capture: {classes}", flush=True)
        print(f"Output directory: ./data/train", flush=True)
        print(f"Camera ID: 0", flush=True)
        print("=" * 60, flush=True)
        
        cap = CaptureImages('./data/train', classes, 0) 
        cap.run(num_images=30)
    except Exception as e:
        import traceback
        print(f"\n{'='*60}", flush=True)
        print(f"FATAL ERROR: {type(e).__name__}", flush=True)
        print(f"Message: {str(e)}", flush=True)
        print(f"{'='*60}", flush=True)
        traceback.print_exc()
        sys.exit(1)
