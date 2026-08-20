# cv_basics.py —— OpenCV 基础图像处理 + 人脸检测实时模糊
# 功能：
#   1. 图像加载与裁剪
#   2. 亮度/对比度增强
#   3. 高斯模糊
#   4. 人脸检测，并对人脸区域自动模糊
# 依赖：pip install opencv-python
import os
import sys
import cv2
import numpy as np

# 预加载 Haar 级联人脸检测器（opencv 自带）
_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def load_image(path):
    """加载图像（BGR）。用 np.fromfile 兼容含中文的路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def save_image(path, img):
    """保存图像，兼容含中文的路径。"""
    ext = os.path.splitext(path)[1] or '.jpg'
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)


def crop_image(img, x0, y0, x1, y1):
    """裁剪矩形区域 [x0:x1, y0:y1]（OpenCV 坐标为 x 列、y 行）。"""
    return img[y0:y1, x0:x1]


def enhance_brightness_contrast(img, alpha=1.4, beta=30):
    """
    亮度/对比度增强：dst = saturate(alpha * src + beta)
    - alpha > 1 提高对比度
    - beta > 0 提高亮度
    """
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def gaussian_blur(img, ksize=(9, 9)):
    """高斯模糊：用高斯核卷积平滑，ksize 越大越模糊。"""
    return cv2.GaussianBlur(img, ksize, 0)


def detect_faces(img):
    """检测图像中的人脸，返回 [(x, y, w, h), ...]。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)


def blur_faces(img, sigma=30):
    """
    人脸区域自动模糊：先检测人脸，再对每个面部矩形做高斯模糊，其余保留。
    返回 (处理结果图, 检测到的人脸数)。
    """
    faces = detect_faces(img)
    out = img.copy()
    for (x, y, w, h) in faces:
        roi = out[y:y + h, x:x + w]
        out[y:y + h, x:x + w] = cv2.GaussianBlur(roi, (0, 0), sigma)
    return out, len(faces)


def process_image(path, out_dir='output'):
    """对单张图片依次演示：裁剪 / 增强 / 高斯模糊 / 人脸模糊，并保存处理前后结果。"""
    img = load_image(path)
    if img is None:
        print(f'无法读取图像: {path}')
        return 0
    os.makedirs(out_dir, exist_ok=True)

    h, w = img.shape[:2]
    cropped = crop_image(img, w // 4, h // 4, 3 * w // 4, 3 * h // 4)   # 取中央 1/4 区域
    enhanced = enhance_brightness_contrast(img)                          # 提亮+加对比
    blurred = gaussian_blur(img)                                         # 整图高斯模糊
    face_blur, n = blur_faces(img)                                       # 仅人脸模糊

    save_image(os.path.join(out_dir, '1_original.png'), img)
    save_image(os.path.join(out_dir, '2_crop.png'), cropped)
    save_image(os.path.join(out_dir, '3_brightness_contrast.png'), enhanced)
    save_image(os.path.join(out_dir, '4_gaussian_blur.png'), blurred)
    save_image(os.path.join(out_dir, '5_face_blur.png'), face_blur)
    print(f'处理完成，检测到 {n} 个人脸，结果已保存到 {out_dir}/')
    return n


def realtime_face_blur():
    """实时人脸模糊：调用摄像头，对每帧检测人脸并模糊，q 退出。"""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('无法打开摄像头')
        return
    print('实时人脸模糊演示中… 按 q 退出')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out, _ = blur_faces(frame)
        cv2.imshow('Realtime Face Blur (q to quit)', out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if '--realtime' in sys.argv:
        realtime_face_blur()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else 'test_input.jpg'
        process_image(path)
