
import time
import cv2


class ImageCapture(object):
    def __init__(self, device, width, height, blur_thres=100):
        self.device = device
        self.width = width
        self.height = height
        self.blur_thres = blur_thres

    @classmethod
    def from_config(cls, config):
        return cls(device=config.IMAGE_CAPTURE_DEVICE,
                   width=config.IMAGE_CAPTURE_WIDTH,
                   height=config.IMAGE_CAPTURE_HEIGHT,
                   blur_thres=config.IMAGE_CAPTURE_BLUR_THRESH)

    def capture(self):
        capture = cv2.VideoCapture(self.device)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not capture.isOpened():
            raise Exception('Failed to open camera capture.')

        for _ in range(0, 10):
            ret, img = capture.read()
            if not ret or self._blur_index(img) < self.blur_thres:
                time.sleep(0.5)
                continue
            capture.release()
            return img

        capture.release()
        raise Exception('Failed to capture image.')

    def _blur_index(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(img_gray, cv2.CV_64F).var()


class FakeImageCapture(object):
    def __init__(self, img_file):
        self.img_file = img_file

    @classmethod
    def from_config(cls, config):
        return cls(img_file=config.DUMMY_IMAGE_FILE)

    def capture(self):
        return cv2.imread(self.img_file)
