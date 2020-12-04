import copy
import cv2
import threading
import queue
from typing import Optional
from vimba import *
import time
from FLSpegtransfer.vision.AlliedVisionUtils import AlliedVisionUtils

# Camera Info.
# /// Camera Name   : GC1290C (Left)
# /// Model Name    : GC1290C (02-2186A)
# /// Camera ID     : DEV_000F31021FD1
# /// Serial Number : 02-2186A-17617
# /// Interface ID  : enp5s0
#
# /// Camera Name   : GC1290C (Right)
# /// Model Name    : GC1290C (02-2186A)
# /// Camera ID     : DEV_000F310199C1
# /// Serial Number : 02-2186A-06108
# /// Interface ID  : enx0050b61b0583


def try_put_frame(q: queue.Queue, cam: Camera, frame: Optional[Frame]):
    try:
        q.put_nowait((cam.get_id(), frame))
    except queue.Full:
        pass


# Thread Objects
class FrameProducer(threading.Thread):
    def __init__(self, cam: Camera, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.cam = cam
        self.frame_queue = frame_queue
        self.killswitch = threading.Event()

    def __call__(self, cam: Camera, frame: Frame):
        # This method is executed within VimbaC context. All incoming frames
        # are reused for later frame acquisition. If a frame shall be queued, the
        # frame must be copied and the copy must be sent, otherwise the acquired
        # frame will be overridden as soon as the frame is reused.
        if frame.get_status() == FrameStatus.Complete:
            if not self.frame_queue.full():
                frame_cpy = copy.deepcopy(frame)
                try_put_frame(self.frame_queue, cam, frame_cpy)
        cam.queue_frame(frame)

    def stop(self):
        self.killswitch.set()

    def setup_camera(self):
        self.cam.get_feature_by_name('GVSPAdjustPacketSize').run()
        self.cam.get_feature_by_name('PixelFormat').set("BGR8Packed")

    def run(self):
        self.log.info('Thread \'FrameProducer({})\' started.'.format(self.cam.get_id()))
        try:
            with self.cam:
                self.setup_camera()
                try:
                    self.cam.start_streaming(self)
                    self.killswitch.wait()
                finally:
                    self.cam.stop_streaming()
        except VimbaCameraError:
            pass
        finally:
            try_put_frame(self.frame_queue, self.cam, None)
        self.log.info('Thread \'FrameProducer({})\' terminated.'.format(self.cam.get_id()))


class FrameConsumer(threading.Thread):
    def __init__(self, frame_queue: queue.Queue):
        threading.Thread.__init__(self)

        self.log = Log.get_instance()
        self.frame_queue = frame_queue
        self.cam_id_left = "DEV_000F31021FD1"
        self.cam_id_right = "DEV_000F310199C1"
        self.img_left = []
        self.img_right = []

    def run(self):
        # KEY_CODE_ENTER = 13
        frames = {}
        alive = True
        self.log.info('Thread \'FrameConsumer\' started.')
        while alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame
                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                for cam_id in sorted(frames.keys()):
                    # cv_image = frames[cam_id].as_opencv_image()
                    np_image = frames[cam_id].as_numpy_ndarray()
                    if cam_id == self.cam_id_left:
                        self.img_left = np_image
                        # cv2.imshow("Left image", cv_image)
                    elif cam_id == self.cam_id_right:
                        self.img_right = np_image
                        # cv2.imshow("Right image", cv_image)

            time.sleep(0.01)
            # Check for shutdown condition
            # if KEY_CODE_ENTER == cv2.waitKey(10):
            #     cv2.destroyAllWindows()
            #     alive = False
        self.log.info('Thread \'FrameConsumer\' terminated.')


class AlliedVisionCapture(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

        # camera variables
        self.av_util = AlliedVisionUtils()

        # threads for capturing
        self.frame_queue = queue.Queue(maxsize=10)
        self.producers = {}
        self.producers_lock = threading.Lock()
        self.consumer = {}
        self.consumer = FrameConsumer(self.frame_queue)
        self.daemon = True
        self.start()

    def __call__(self, cam: Camera, event: CameraEvent):
        # New camera was detected. Create FrameProducer, add it to active FrameProducers
        if event == CameraEvent.Detected:
            with self.producers_lock:
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)
                self.producers[cam.get_id()].start()

        # An existing camera was disconnected, stop associated FrameProducer.
        elif event == CameraEvent.Missing:
            with self.producers_lock:
                producer = self.producers.pop(cam.get_id())
                producer.stop()
                producer.join()

    def run(self):
        log = Log.get_instance()
        vimba = Vimba.get_instance()
        vimba.enable_log(LOG_CONFIG_INFO_CONSOLE_ONLY)

        log.info('Thread \'MainThread\' started.')
        with vimba:
            # Construct FrameProducer threads for all detected cameras
            for cam in vimba.get_all_cameras():
                self.producers[cam.get_id()] = FrameProducer(cam, self.frame_queue)

            # Start FrameProducer threads
            with self.producers_lock:
                for producer in self.producers.values():
                    producer.start()

            # Start and wait for consumer to terminate
            vimba.register_camera_change_handler(self)
            self.consumer.start()
            self.consumer.join()
            vimba.unregister_camera_change_handler(self)

            # Stop all FrameProducer threads
            with self.producers_lock:
                # Initiate concurrent shutdown
                for producer in self.producers.values():
                    producer.stop()

                # Wait for shutdown to complete
                for producer in self.producers.values():
                    producer.join()
        log.info('Thread \'MainThread\' terminated.')

    def capture(self, which='original'):
        if which=='original':
            return self.consumer.img_left, self.consumer.img_right
        elif which=='rectified':
            if len(self.consumer.img_left) == 0 or len(self.consumer.img_right) == 0:
                return [], []
            else:
                return self.av_util.rectify(self.consumer.img_left, self.consumer.img_right)


if __name__ == "__main__":
    av = AlliedVisionCapture()
    import numpy as np
    while True:
        img_left, img_right = av.capture()
        if len(img_left) == 0 or len(img_right) == 0:
            pass
        else:
            scale = 0.6  # percent of original size
            w = int(img_left.shape[1] * scale)
            h = int(img_left.shape[0] * scale)
            dim = (w,h)
            img_left_resized = cv2.resize(img_left, dim, interpolation=cv2.INTER_AREA)
            img_right_resized = cv2.resize(img_right, dim, interpolation=cv2.INTER_AREA)
            img_stacked = np.concatenate((img_left_resized, img_right_resized), axis=1)
            cv2.imshow("stereo_image", img_stacked)
            cv2.waitKey(1)