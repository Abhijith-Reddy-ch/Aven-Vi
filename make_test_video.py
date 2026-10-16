import cv2
import numpy as np

base_img = cv2.imread('assets/visionfusion_demo.png')
h, w = base_img.shape[:2]
fps = 30
num_frames = 90  # 3 seconds

out = cv2.VideoWriter('driving_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Crop the prominent car on the right
# Box in demo: [1314, 667, 1910, 1032]
car_roi = base_img[667:1032, 1314:1910].copy()
car_h, car_w = car_roi.shape[:2]

for i in range(num_frames):
    frame = base_img.copy()
    # Simulate an approaching vehicle in the center lane
    # Scale grows from 0.4 to 1.1, Y position moves down from 680 to 750
    progress = i / float(num_frames)
    scale = 0.35 + 0.65 * progress
    nw, nh = int(car_w * scale), int(car_h * scale)
    if nw > 0 and nh > 0:
        car_resized = cv2.resize(car_roi, (nw, nh))
        cx = int(w * 0.5)
        cy = int(680 + 70 * progress)
        x1 = max(0, cx - nw // 2)
        y1 = max(0, cy - nh // 2)
        x2 = min(w, x1 + nw)
        y2 = min(h, y1 + nh)
        
        # Alpha blend car into center
        car_patch = car_resized[:(y2 - y1), :(x2 - x1)]
        frame[y1:y2, x1:x2] = cv2.addWeighted(frame[y1:y2, x1:x2], 0.1, car_patch, 0.9, 0)
    
    out.write(frame)

out.release()
print(f'Generated driving_video.mp4: {w}x{h}, {num_frames} frames at {fps} fps')
