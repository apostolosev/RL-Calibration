import os
import shutil
import time

import cv2 as cv


def millis():
    return round(time.time() * 1000)


def create_dir(folder, remove=True):
    if remove:
        try:
            shutil.rmtree(folder)
        except FileNotFoundError:
            pass
        os.makedirs(folder, exist_ok=True)


class ImageGrabber:
    def __init__(self, cap_id=0):
        self.cap = cv.VideoCapture(cap_id)

    def __del__(self):
        self.cap.release()

    def info(self):
        height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        fps = self.cap.get(cv.CAP_PROP_FPS)
        exposure = self.cap.get(cv.CAP_PROP_EXPOSURE)
        brightness = self.cap.get(cv.CAP_PROP_BRIGHTNESS)
        contrast = self.cap.get(cv.CAP_PROP_CONTRAST)
        saturation = self.cap.get(cv.CAP_PROP_SATURATION)
        hue = self.cap.get(cv.CAP_PROP_HUE)
        gain = self.cap.get(cv.CAP_PROP_GAIN)

        print("---------------- Webcam parameter info ----------------")
        print("Inage height: {}".format(height))
        print("Image width: {}".format(width))
        print("Frame rate: {}".format(fps))
        print("Exposure: {}".format(exposure))
        print("Brightness: {}".format(brightness))
        print("Contrast: {}".format(contrast))
        print("Saturation: {}".format(saturation))
        print("Hue: {}".format(hue))
        print("Gain: {}".format(gain))
        print("-------------------------------------------------------")

    def set_resolution(self, height=1280, width=720):
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, width)

    def set_frame_rate(self, fps=25):
        self.cap.set(cv.CAP_PROP_FPS, fps)

    def set_exposure(self, exposure=-1):
        self.cap.set(cv.CAP_PROP_EXPOSURE, exposure)

    def show(self):
        prev_time = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            new_time = time.time()
            fps = 1/(new_time - prev_time)
            prev_time = new_time
            if ret:
                cv.imshow("frame", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break
        cv.destroyAllWindows()

    def capture(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv.imshow("Frame", frame)
                if cv.waitKey(1) & 0xFF == ord("c"):
                    cv.imwrite("data/calibration/webcam.bmp", frame)
                    print("Captured image webcam.jpg")
                    break
            else:
                break
        cv.destroyAllWindows()

    def capture_stream(self):
        frame_count = 0
        warm_up_time = 5000
        print("Warming up...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if frame_count == 0:
                init_pos_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)
            frame_count += 1
            pos_msec = self.cap.get(cv.CAP_PROP_POS_MSEC)
            if pos_msec - init_pos_msec > warm_up_time:
                print("Finished")
                print("-------------------------------------------------------")
                break
        print("Recording video stream...")
        print("Press \"c\" to stat recording and \"q\" to stop")
        fname = "stream_" + str(millis())
        create_dir(fname)
        start_recording = False
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if cv.waitKey(1) & 0xFF == ord("c"):
                start_recording = True
                print("Started recording")
            if ret:
                cv.imshow("frame", frame)
                if start_recording:
                    cv.imwrite(fname + "/webcam_" + str(millis()) + ".jpg", frame)
                if cv.waitKey(1) & 0xFF == ord("q"):
                    print("Finished")
                    print("-------------------------------------------------------")
                    break


def main():
    grabber = ImageGrabber()
    grabber.set_resolution(height=720, width=960)
    grabber.info()
    grabber.capture()


if __name__ == "__main__":
    main()
