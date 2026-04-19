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
base_model = DETR(num_classes=22)
checkpoint = torch.load("D:\\NCKH\\SIGN-VSL\\NCKH-SIGN-VSL\\model_test\\best_signdetr_model_base_v2.pth", map_location='cpu')

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    base_model.load_state_dict(checkpoint['model_state_dict'])
else:
    base_model.load_state_dict(checkpoint)

base_model.eval()

# Bọc nó lại
wrapped_model = DETRMobileWrapper(base_model)
wrapped_model.eval()

# 3. Tracing (Tạo bản đồ đường đi của dữ liệu)
print("Đang Tracing model (Sẽ mất vài phút)...")
example_input = torch.rand(1, 3, 512, 512) 

# ĐÃ SỬA: Thêm check_trace=False và strict=False vào đây!
traced_script_module = torch.jit.trace(wrapped_model, example_input, check_trace=False, strict=False)

# 4. Tối ưu hóa cho Mobile (Cực kỳ quan trọng)
print("Đang tối ưu hóa cho Mobile...")
optimized_traced_model = optimize_for_mobile(traced_script_module)

# 5. Xuất xưởng!
optimized_traced_model._save_for_lite_interpreter("signdetr_mobile.ptl")
print("✅ Hoàn tất! Model đã được nén thành: signdetr_mobile.ptl")