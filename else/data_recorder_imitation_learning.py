import time, threading
import cv2
import numpy as np
import tkinter as tk
import os
from vision.AlliedVisionCapture import AlliedVisionCapture
from vision.ZividCapture import ZividCapture
from motion.dvrkDualArm import dvrkDualArm


class DataRecorder():
	def __init__(self):		
		# Variables		
		self.collecting = False
		self.trial_num = -1
		self.i = 0
		self.robot_states = []

		# Instances
		self.zividM = ZividCapture(which_camera="overhead")
		self.zividS = ZividCapture(which_camera="inclined")
		self.zividM.start()
		self.zividS.start()
				
		self.thread = threading.Thread(target=self.loop)
		self.thread.daemon = True
		self.dvrk = dvrkDualArm()		
		self.startup()

	def startup(self):
		top = tk.Tk()
		top.title('IL GUI')
		top.geometry('400x200')
		B1 = tk.Button(top, text="Start Trial", command=self.collect)
		B2 = tk.Button(top, text="Stop Trial", command=self.stop)
		B1.pack()
		B2.pack()
		self.thread.start()
		top.mainloop()

	def collect(self):
		self.trial_num += 1
		self.i = 0
		print ("starting trial: ", self.trial_num)
		dir = "data_recorded/" + str(self.trial_num)
		if not os.path.exists(dir):
			os.makedirs(dir)
		self.collecting = True

	def stop(self):
		print ("stopping trial: ", self.trial_num)
		self.collecting = False

	def loop(self):
		while True:
			if self.collecting:
				print (str(self.i), "images saved")
					
				# capture images
				colorM, depthM, _ = self.zividM.capture_3Dimage(color='BGR')
				colorS, depthS, _ = self.zividS.capture_3Dimage(color='BGR')

				# capture robot states
				(pos, rot, jaw), _ = self.dvrk.get_pose()
				joint, _ = self.dvrk.get_joint()
				robot_state = [pos, rot, jaw, joint]
				
				# save
				dir = "data_recorded/" + str(self.trial_num)
				np.save(dir + '/colorM' + str(self.i), colorM)
				np.save(dir + '/colorS' + str(self.i), colorS)
				np.save(dir + '/depthM' + str(self.i), depthM)
				np.save(dir + '/depthS' + str(self.i), depthS)
				np.save(dir + '/positions' + str(self.i), robot_state)
				
				# print ("robot_state: ", robot_state)				
				self.i += 1
				cv2.imshow("", colorS)
				cv2.waitKey(1)

if __name__ == "__main__":
	record = DataRecorder()
