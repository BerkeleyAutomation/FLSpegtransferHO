# FLSpegtransfer

# Manual Dependency Install

Parts of the installation build tools:

    sudo apt install build-essential

If you don't have ROS installed already, follow these instructions to install it:  [http://wiki.ros.org/noetic/Installation]

    source /opt/ros/noetic/setup.bash

Setup and activate a virtualenv:

    virtualenv venv
    . venv/bin/activate
    pip install probreg==0.3.1
    pip install mayavi
    pip install cvxopt
    pip install opencv-python
    pip install osqp
