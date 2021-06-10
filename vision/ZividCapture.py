import zivid
import numpy as np
import datetime


class ZividCapture():
    def __init__(self, which_camera="inclined"):
        # data members
        self.which_camera = which_camera
        if which_camera == "inclined":
            self.serial_number = "20077B66"
        elif which_camera == "overhead":
            self.serial_number = "19163962"
        self.camera = None

        # measuring frame rate
        self.t = 0.0
        self.t_prev = 0.0
        self.interval = 0.0
        self.fps = 0.0

    def start(self):
        # pass
        self.configure_setting()
        app = zivid.Application()
        self.camera = app.connect_camera(serial_number=self.serial_number)
        print("Zivid initialized")

    def configure_setting(self):
        # 2D image setting
        self.settings_2d = zivid.Settings2D()
        self.settings_2d.acquisitions.append(zivid.Settings2D.Acquisition())
        self.settings_2d.acquisitions[0].aperture = 4.76
        self.settings_2d.acquisitions[0].exposure_time = datetime.timedelta(microseconds=10000)
        self.settings_2d.brightness = 1.3

        # 3D capture setting
        self.settings = zivid.Settings()
        self.settings.acquisitions.append(zivid.Settings.Acquisition())
        self.settings.acquisitions[0].aperture = 4.76
        self.settings.acquisitions[0].exposure_time = datetime.timedelta(microseconds=10000)
        self.settings.brightness = 1.3
        self.settings.processing.filters.outlier.removal.enabled = True
        self.settings.processing.filters.outlier.removal.threshold = 5.0
        self.settings.processing.filters.noise.removal.enabled = False
        self.settings.processing.filters.outlier.removal.enabled = True
        self.settings.processing.Filters.Smoothing.Gaussian.enabled = True
        self.settings.processing.Filters.Smoothing.Gaussian.sigma = 1.5

    def capture_2Dimage(self, color='RGB'):      # measured as 20~90 fps
        with self.camera.capture(self.settings_2d) as frame_2d:
            np_array = frame_2d.image_rgba().copy_data()
            if color == 'BGR':
                np_array[:, :, [0, 2]] = np_array[:, :, [2, 0]]
            return np_array[:, :, :3]

    def capture_3Dimage(self, color='RGB'):      # measured as 7~10 fps
        with self.camera.capture(self.settings) as frame:
            np_array = frame.point_cloud().copy_data("xyzrgba")
            if color=='RGB':
                img_color = np.dstack([np_array["r"], np_array["g"], np_array["b"]])   # image data
            elif color == 'BGR':
                img_color = np.dstack([np_array["b"], np_array["g"], np_array["r"]])  # image data
            img_point = np.dstack([np_array["x"], np_array["y"], np_array["z"]])   # pcl data in (mm)
            img_depth = img_point[:,:,2]  # depth data in (mm)
            return img_color, img_depth, img_point


if __name__ == "__main__":
    import cv2
    zc = ZividCapture(which_camera='inclined')
    zc.start()
    image = zc.capture_2Dimage(color='BGR')
    cv2.imshow("", image)
    cv2.waitKey(0)