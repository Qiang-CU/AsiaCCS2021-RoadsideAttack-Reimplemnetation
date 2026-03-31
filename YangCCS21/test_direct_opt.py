"""
Direct adversarial point optimization (no mesh).
Tests whether directly optimizing point coordinates can push RPN logits up.
"""
import os, sys, torch, yaml, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.pointrcnn_wrapper import PointRCNNWrapper
from utils.kitti_utils import KITTIDataset
from attack.inject import build_bev_occupancy, sample_injection_position, inject_points

device = 'cuda:0'
torch.cuda.set_device(0)

with open('configs/attack_config.yaml') as f:
    config = yaml.safe_load(f)

wrapper = PointRCNNWrapper(
    config['model']['pointrcnn_config'],
    config['model']['pointrcnn_ckpt'],
    device=device, enable_ste=False,
)

dataset = KITTIDataset(config['data']['kitti_root'], split='val',
                       pc_range=config['data']['pc_range'])

sample = dataset[4]
inj_pos = np.array([8.25, -8.25, 0.0])
pc_np = sample['pointcloud']

# Create adversarial points directly (no mesh/render):
# 200 points in a car-like shape near injection
rng = np.random.RandomState(42)
adv_base = np.zeros((200, 3), dtype=np.float32)
adv_base[:, 0] = rng.uniform(-1.5, 1.5, 200)  # length
adv_base[:, 1] = rng.uniform(-0.7, 0.7, 200)  # width
adv_base[:, 2] = rng.uniform(-0.3, 1.2, 200)   # height
adv_base += inj_pos

adv_pts = torch.tensor(adv_base, dtype=torch.float32, device=device, requires_grad=True)
optimizer = torch.optim.Adam([adv_pts], lr=0.01)

print("Direct point optimization (no mesh):")
print(f"{'step':>5} | {'loss':>8} | {'logit':>8} | {'fg_score':>8} | {'grad':>8} | {'#prop':>5}")
print("-" * 70)

for step in range(500):
    optimizer.zero_grad()
    
    pc_tensor = torch.tensor(pc_np, dtype=torch.float32, device=device)
    adv_4 = torch.cat([adv_pts, torch.ones(200, 1, device=device)], dim=1)
    merged = torch.cat([pc_tensor, adv_4], dim=0)
    n_scene = pc_tensor.shape[0]
    n_adv = 200
    
    result = wrapper.forward_with_grad(merged, rpn_only=True, run_post=False)
    
    logits = result.get('point_cls_logits')
    scores = result.get('point_cls_scores')
    
    if logits is not None:
        adv_logits = logits[n_scene:n_scene + n_adv]
        adv_scores = scores[n_scene:n_scene + n_adv]
        loss = -adv_logits.mean()
        loss.backward()
        
        gn = adv_pts.grad.norm().item() if adv_pts.grad is not None else 0
        optimizer.step()
        
        if step % 25 == 0 or step == 499:
            print(f"{step:5d} | {loss.item():8.4f} | {adv_logits.mean().item():8.4f} | "
                  f"{adv_scores.mean().item():8.4f} | {gn:8.4f} | -")
    else:
        print(f"{step:5d} | no logits")

wrapper.remove_hook()
print("\nDone.")
