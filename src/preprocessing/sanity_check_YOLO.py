import cv2

# مسیر یکی از عکس ها و لیبل های ساخته شده را بدهید
img_path = "Data/processed/yolo_fracture/images/train/1196_z19.jpg"
txt_path = "Data/processed/yolo_fracture/labels/train/1196_z19.txt"

img = cv2.imread(img_path)
h, w, _ = img.shape

with open(txt_path, 'r') as f:
    for line in f.readlines():
        class_id, cx, cy, bw, bh = map(float, line.strip().split())
        
        # تبدیل دوباره به پیکسل
        x_center, y_center = int(cx * w), int(cy * h)
        box_w, box_h = int(bw * w), int(bh * h)
        
        x_min = int(x_center - (box_w / 2))
        y_min = int(y_center - (box_h / 2))
        
        cv2.rectangle(img, (x_min, y_min), (x_min + box_w, y_min + box_h), (0, 255, 0), 2)

cv2.imshow("Check BBox", img)
cv2.waitKey(0)