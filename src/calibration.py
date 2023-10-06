import cv2 as cv
import numpy as np

from capture import ImageGrabber

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def shape_approximation(mask):
    mask = 255 * mask.numpy().astype(np.uint8)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    pts = np.zeros((4, 2))
    if len(contours) == 1:
        rbox = cv.minAreaRect(*contours)
        pts = np.array(cv.boxPoints(rbox).astype(np.float32))
    elif len(contours) > 1:
        contours = sorted(contours, key=lambda x: -cv.contourArea(x))
        rbox = cv.minAreaRect(contours[0])
        pts = np.array(cv.boxPoints(rbox).astype(np.float32))
    return order_points(pts)


class Homography:
    def __init__(self, width=1, height=1, corners=None):
        self.image = None
        self.warped = None
        self.copy = None
        self.scaled = None
        self.width = width
        self.height = height
        self.corners = corners
        self.M = None
        self.S = None

    def set_frame(self, frame):
        self.image = frame

    def click_event(self, event, x, y, flags, params):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(self.copy, center=(x, y), radius=2, color=(0, 0, 255), thickness=-1)
            cv.imshow("Image", self.copy)
            cv.waitKey(1)
            self.corners.append((x, y))

    def get_corners(self, select=0):
        self.corners = []
        if select == 0:
            self.copy = self.image.copy()
        elif select == 1:
            self.copy = self.warped.copy()
        elif select == 2:
            self.copy = self.scaled.copy()
        cv.imshow("Image", self.copy)
        cv.setMouseCallback("Image", self.click_event)
        cv.waitKey(0)
        self.corners = np.array(self.corners, dtype=np.float32)
        return self.corners

    def loadM(self, path):
        self.M = np.load(path)
        self.warped = cv.warpPerspective(self.image, self.M, (self.image.shape[1], self.image.shape[0]))

    def loadS(self, path):
        self.S = np.load(path)
        sx = self.S[0, 0]
        sy = self.S[1, 1]
        self.scaled = cv.warpPerspective(self.warped, self.S, (int(self.image.shape[1] * sx), int(self.image.shape[0] * sy)))

    def warp(self, corners=None):
        self.corners = self.get_corners() if corners is None else corners
        (topLeft, topRight, bottomRight, bottomLeft) = self.corners
        widthA = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
        widthB = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
        heightB = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        dst = np.array([[0, 0],
                        [maxWidth, 0],
                        [maxWidth, maxHeight],
                        [0, maxHeight]
                        ], dtype=np.float32)
        self.M, _ = cv.findHomography(self.corners, dst)
        np.save("data/calibration/warp.npy", self.M)
        self.warped = cv.warpPerspective(self.image, self.M, (self.image.shape[1], self.image.shape[0]))
        cv.imshow("Warped", self.warped)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.M

    def scale(self, corners=None):
        self.corners = self.get_corners(select=1) if corners is None else corners
        (topLeft, topRight, bottomRight, bottomLeft) = self.corners
        widthA = np.sqrt(((bottomRight[0] - bottomLeft[0]) ** 2) + ((bottomRight[1] - bottomLeft[1]) ** 2))
        widthB = np.sqrt(((topRight[0] - topLeft[0]) ** 2) + ((topRight[1] - topLeft[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        heightA = np.sqrt(((topRight[0] - bottomRight[0]) ** 2) + ((topRight[1] - bottomRight[1]) ** 2))
        heightB = np.sqrt(((topLeft[0] - bottomLeft[0]) ** 2) + ((topLeft[1] - bottomLeft[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        sx = self.width / maxWidth
        sy = self.height / maxHeight
        self.S = np.array([[sx, 0, 0],
                           [0, sy, 0],
                           [0, 0, 1]], dtype=np.float32)
        np.save("data/calibration/scale.npy", self.S)
        self.scaled = cv.warpPerspective(self.warped, self.S, (int(self.image.shape[1] * sx), int(self.image.shape[0] * sy)))
        cv.imshow("Scaled", self.scaled)
        cv.waitKey(0)
        cv.destroyAllWindows()
        return self.S

    def distance(self, corners=None):
        self.corners = self.get_corners(select=2) if corners is None else corners
        p1 = self.corners[0, :]
        p2 = self.corners[1, :]
        return np.linalg.norm(p1 - p2)


def main():
    # Grab an image
    # grabber = ImageGrabber()
    # grabber.set_resolution(height=720, width=960)
    # grabber.info()
    # grabber.capture()

    # Compute the calibration matrix
    pCorr = Homography(height=89, width=107)
    pCorr.set_frame(cv.imread("data/calibration/webcam.bmp"))
    pCorr.warp()
    pCorr.scale()


if __name__ == "__main__":
    main()


