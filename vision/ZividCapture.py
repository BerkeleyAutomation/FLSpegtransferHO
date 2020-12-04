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
        self.settings = zivid.Settings()
        self.settings_2d = zivid.Settings2D()
        self.camera = []

        # measuring frame rate
        self.t = 0.0
        self.t_prev = 0.0
        self.interval = 0.0
        self.fps = 0.0

    def start(self):
        self.configure_setting()
        app = zivid.Application()
        self.camera = app.connect_camera(serial_number=self.serial_number, settings=self.settings)
        print ("Zivid initialized")

    def configure_setting(self):
        # 2D image setting
        self.settings_2d.iris = 22
        self.settings_2d.exposure_time = datetime.timedelta(microseconds=8333)

        # 3D capture setting
        self.settings.exposure_time = datetime.timedelta(microseconds=10000)
        self.settings.iris = 22
        self.settings.brightness = 1.0
        self.settings.gain = 2.00
        self.settings.bidirectional = False
        # self.settings.filters.contrast.enabled = True
        # self.settings.filters.contrast.threshold = 5
        # self.settings.filters.gaussian.enabled = True
        # self.settings.filters.gaussian.sigma = 1.5
        # self.settings.filters.outlier.enabled = True
        # self.settings.filters.outlier.threshold = 1.0
        # self.settings.filters.reflection.enabled = False
        # self.settings.filters.saturated.enabled = True
        self.settings.blue_balance = 1.08
        self.settings.red_balance = 1.71

    def capture_2Dimage(self, color='RGB'):      # measured as 20~90 fps
        with self.camera.capture_2d(self.settings_2d) as frame_2d:
            np_arra = frame_2d.image().to_array()
            # print(np_array.dtype.names)
            if color == 'RGB':
                image = np.dstack([np_array["r"], np_array["g"], np_array["b"]])  # image data
            elif color == 'BGR':
                image = np.dstack([np_array["b"], np_array["g"], np_array["r"]])  # image data
            return image

    def capture_3Dimage(self, color='RGB'):      # measured as 7~10 fps
        with self.camera.capture(settings_collection=[self.settings]) as frame:
            np_array = frame.get_point_cloud().to_array()
            # print (np_array.dtype.names)
            if color=='RGB':
                img_color = np.dstack([np_array["r"], np_array["g"], np_array["b"]])   # image data
            elif color == 'BGR':
                img_color = np.dstack([np_array["b"], np_array["g"], np_array["r"]])  # image data
            img_point = np.dstack([np_array["x"], np_array["y"], np_array["z"]])   # pcl data in (mm)
            img_depth = img_point[:,:,2]  # depth data in (mm)
            return img_color, img_depth, img_point