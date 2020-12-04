import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy

Gst.init(None)


image_arr1 = None
image_arr2 = None

def gst_to_opencv(sample):
    buf = sample.get_buffer()
    caps = sample.get_caps()
    print(caps.get_structure(0).get_value('format'))
    print(caps.get_structure(0).get_value('height'))
    print(caps.get_structure(0).get_value('width'))
    print(buf.get_size())

    arr = numpy.ndarray(
        (caps.get_structure(0).get_value('height'),
         caps.get_structure(0).get_value('width'),
         3),
        buffer=buf.extract_dup(0, buf.get_size()),
        dtype=numpy.uint8)
    return arr

def new_buffer1(sink, data):
    global image_arr1
    sample = sink.emit("pull-sample")
    # buf = sample.get_buffer()
    # print "Timestamp: ", buf.pts
    arr = gst_to_opencv(sample)
    image_arr1 = arr
    return Gst.FlowReturn.OK

def new_buffer2(sink, data):
    global image_arr2
    sample = sink.emit("pull-sample")
    # buf = sample.get_buffer()
    # print "Timestamp: ", buf.pts
    arr = gst_to_opencv(sample)
    image_arr2 = arr
    return Gst.FlowReturn.OK


# Create the empty pipeline
pipeline = Gst.Pipeline.new("test-pipeline")

# For left images
# Create the elements
source1 = Gst.ElementFactory.make("decklinkvideosrc", None)
convert1 = Gst.ElementFactory.make("videoconvert", None)
sink1 = Gst.ElementFactory.make("appsink", None)
caps1 = Gst.caps_from_string("video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}")

if not source1 or not sink1 or not pipeline:
    print("Not all elements could be created.")
    exit(-1)

source1.set_property("mode", 0)
source1.set_property("connection", 0)        # SDI
source1.set_property("device-number", 0)     # Deice number
sink1.set_property("emit-signals", True)
sink1.set_property("caps", caps1)
sink1.connect("new-sample", new_buffer1, sink1)

# Build the pipeline
pipeline.add(source1)
pipeline.add(convert1)
pipeline.add(sink1)

# For right images
# Create the elements
source2 = Gst.ElementFactory.make("decklinkvideosrc", None)
convert2 = Gst.ElementFactory.make("videoconvert", None)
sink2 = Gst.ElementFactory.make("appsink", None)
caps2 = Gst.caps_from_string("video/x-raw, format=(string){BGR, GRAY8}; video/x-bayer,format=(string){rggb,bggr,grbg,gbrg}")

if not source2 or not sink2 or not pipeline:
    print("Not all elements could be created.")
    exit(-1)

source2.set_property("mode", 0)
source2.set_property("connection", 0)        # SDI
source2.set_property("device-number", 1)     # Deice number
sink2.set_property("emit-signals", True)
sink2.set_property("caps", caps2)
sink2.connect("new-sample", new_buffer2, sink2)

# import pdb; pdb.set_trace()
# Build the pipeline
pipeline.add(source2)
pipeline.add(convert2)
pipeline.add(sink2)

# Link
# source1.link(convert1)
# convert1.link(sink1)
# source2.link(convert2)
# convert2.link(sink2)
if not Gst.Element.link(source1, convert1):
    print("Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(convert1, sink1):
    print("Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(source2, convert2):
    print("Elements could not be linked.")
    exit(-1)

if not Gst.Element.link(convert2, sink2):
    print("Elements could not be linked.")
    exit(-1)

# Start playing
ret = pipeline.set_state(Gst.State.PLAYING)
if ret == Gst.StateChangeReturn.FAILURE:
    print("Unable to set the pipeline to the playing state.")
    exit(-1)

# Wait until error or EOS
bus = pipeline.get_bus()

# Parse message
while True:
    message = bus.timed_pop_filtered(10000, Gst.MessageType.ANY)
    # print "image_arr: ", image_arr
    if image_arr1 is not None:
        cv2.imshow("appsink image arr1", image_arr1)
        cv2.waitKey(1)
    if image_arr2 is not None:
        cv2.imshow("appsink image arr2", image_arr2)
        cv2.waitKey(1)
    if message:
        if message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(("Error received from element %s: %s" % (
                message.src.get_name(), err)))
            print(("Debugging information: %s" % debug))
            break
        elif message.type == Gst.MessageType.EOS:
            print("End-Of-Stream reached.")
            break
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                print(("Pipeline state changed from %s to %s." %
                       (old_state.value_nick, new_state.value_nick)))
        else:
            print("Unexpected message received.")

# Free resources
pipeline.set_state(Gst.State.NULL)