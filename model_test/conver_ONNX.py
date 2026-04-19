import torch
from torch.utils.mobile_optimizer import optimize_for_mobile
from model_resnet import DETR

# 1. Tạo lớp Vỏ bọc để gỡ cái Dictionary ra thành Tuple
class DETRMobileWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Truyền ảnh vào model gốc
        out = self.model(x)
        # Bóc tách và chỉ trả về 2 cục Tensor thuần túy để Flutter dễ đọc
        return out['pred_logits'], out['pred_boxes']

# 2. Load model cũ của ông lên
print("Đang load model...")
model = DETR(num_classes=22)
checkpoint = torch.load("D:\\NCKH\\SIGN-VSL\\NCKH-SIGN-VSL\\model_test\\best_signdetr_model_base_v2.pth", map_location='cpu')
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

# 1. Load model và set eval
# (Nhớ dùng cái DETRMobileWrapper trả về 2 cục Tensor như Cách 2 để dễ convert)
model.eval()

# 2. Tạo ảnh nháp 512x512
dummy_input = torch.randn(1, 3, 512, 512, device='cpu')

# 3. Xuất ra ONNX (Dùng opset_version cao nhất có thể, thường là 16 hoặc 17)
torch.onnx.export(
    model, 
    dummy_input, 
    "signdetr.onnx",
    export_params=True,
    opset_version=16, # Cực kỳ quan trọng để hỗ trợ Transformer
    do_constant_folding=True,
    input_names=['input'],
    output_names=['pred_logits', 'pred_boxes'],
)
print("Xuất ONNX thành công!")