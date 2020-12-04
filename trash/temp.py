import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from gi.repository import GObject

GObject.threads_init()
Gst.init(None)

import sys, os

class VideoBox():
    def __init__(self):
        mainloop = GObject.MainLoop()
        # Create transcoding pipeline
        self.pipeline = Gst.Pipeline()

        self.v4lsrc1 = Gst.ElementFactory.make('v4l2src', None)
        self.v4lsrc1.set_property("device", "/dev/video0")
        self.pipeline.add(self.v4lsrc1)

        self.v4lsrc2 = Gst.ElementFactory.make('v4l2src', None)
        self.v4lsrc2.set_property("device", "/dev/video1")
        self.pipeline.add(self.v4lsrc2)

        camera1caps = Gst.Caps.from_string("video/x-raw, width=320,height=240")
        self.camerafilter1 = Gst.ElementFactory.make("capsfilter", "filter1")
        self.camerafilter1.set_property("caps", camera1caps)
        self.pipeline.add(self.camerafilter1)

        self.videoenc = Gst.ElementFactory.make("theoraenc", None)
        self.pipeline.add(self.videoenc)

        self.videodec = Gst.ElementFactory.make("theoradec", None)
        self.pipeline.add(self.videodec)

        self.videortppay = Gst.ElementFactory.make("rtptheorapay", None)
        self.pipeline.add(self.videortppay)

        self.videortpdepay = Gst.ElementFactory.make("rtptheoradepay", None)
        self.pipeline.add(self.videortpdepay)

        self.textoverlay = Gst.ElementFactory.make("textoverlay", None)
        self.textoverlay.set_property("text", "Talk is being recorded")
        self.pipeline.add(self.textoverlay)


        camera2caps = Gst.Caps.from_string("video/x-raw, width=320,height=240")
        self.camerafilter2 = Gst.ElementFactory.make("capsfilter", "filter2")
        self.camerafilter2.set_property("caps", camera2caps)
        self.pipeline.add(self.camerafilter2)



        self.videomixer = Gst.ElementFactory.make('videomixer', None)
        self.pipeline.add(self.videomixer)

        self.videobox1 = Gst.ElementFactory.make('videobox', None)
        self.videobox1.set_property("border-alpha", 0)
        self.videobox1.set_property("top", 0)
        self.videobox1.set_property("left", -320)
        self.pipeline.add(self.videobox1)

        self.videoformatconverter1 = Gst.ElementFactory.make('videoconvert', None)
        self.pipeline.add(self.videoformatconverter1)

        self.videoformatconverter2 = Gst.ElementFactory.make('videoconvert', None)
        self.pipeline.add(self.videoformatconverter2)

        self.videoformatconverter3 = Gst.ElementFactory.make('videoconvert', None)
        self.pipeline.add(self.videoformatconverter3)

        self.videoformatconverter4 = Gst.ElementFactory.make('videoconvert', None)
        self.pipeline.add(self.videoformatconverter4)

        self.xvimagesink = Gst.ElementFactory.make('xvimagesink', None)
        self.pipeline.add(self.xvimagesink)

        self.v4lsrc1.link(self.camerafilter1)
        self.camerafilter1.link(self.videoformatconverter1)
        self.videoformatconverter1.link(self.textoverlay)
        self.textoverlay.link(self.videobox1)
        self.videobox1.link(self.videomixer)

        self.v4lsrc2.link(self.camerafilter2)
        self.camerafilter2.link(self.videoformatconverter2)
        self.videoformatconverter2.link(self.videoenc)
        self.videoenc.link(self.videortppay)
        self.videortppay.link(self.videortpdepay)
        self.videortpdepay.link(self.videodec)
        self.videodec.link(self.videoformatconverter3)
        self.videoformatconverter3.link(self.videomixer)

        self.videomixer.link(self.videoformatconverter4)
        self.videoformatconverter4.link(self.xvimagesink)
        self.pipeline.set_state(Gst.State.PLAYING)
        mainloop.run()


if __name__ == "__main__":
    app = VideoBox()
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    exit_status = app.run(sys.argv)
    sys.exit(exit_status)