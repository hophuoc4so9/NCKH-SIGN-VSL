import tensorflow as tf

# Load thư mục TensorFlow SavedModel vừa tạo
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model_detr')

# Ép nó tối ưu hóa (Lượng hóa - Quantization để model nhẹ đi một nửa)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Thực hiện convert
tflite_model = converter.convert()

# Lưu thành file .tflite
with open('signdetr.tflite', 'wb') as f:
    f.write(tflite_model)
