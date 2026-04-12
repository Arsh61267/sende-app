import torch
import numpy as np
import cv2

# =========================
# SIMPLE GRADCAM — SAFE FINAL
# =========================

class SimpleGradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self._save_activation)

        try:
            target_layer.register_full_backward_hook(self._save_gradient)
        except:
            target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):

        self.model.zero_grad()

        out = self.model(input_tensor)
        score = out.sum()
        score.backward()

        grads = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = (grads * self.activations).sum(dim=1).squeeze()

        cam = torch.relu(cam)
        cam = cam.cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() + 1e-6)

        return cam


# =========================
# OVERLAY CAM — DISPLAY SAFE
# =========================

def overlay_cam(img_np, cam):

    cam = np.clip(cam, 0, 1)

    h, w = img_np.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heat = cv2.applyColorMap(
        np.uint8(255 * cam_resized),
        cv2.COLORMAP_JET
    )

    if img_np.dtype != np.uint8:
        img_np = img_np.astype(np.uint8)

    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)

    if heat.shape[:2] != img_np.shape[:2]:
        heat = cv2.resize(heat, (img_np.shape[1], img_np.shape[0]))

    overlay = cv2.addWeighted(
        img_np, 0.6,
        heat, 0.4,
        0
    )

    return overlay
