import numpy as np
import cv2
import ipdb 
import os
import glob
import csv
import matplotlib.pyplot as plt
import sys

'''
INPUT: Image of the rover with marked centroid (by algo). 
	   Range at which that image was captured (set a default value incase range is not received)
DISPLAY: Intermediate results for the video (to be shown in demonstration)
Output: Percentage of correct detection

Receive image inputs with centroid coorfinates determined by the algorithm. The program displays the image on the screen.
Use the point and click code to identify the true centroid of the rover.
Set a threshold of euclidian distance between these two rovers (as a function of range). If the euclidian distanc eis above a certain threshold mark it as 
incorrectly detected else as correctly detected. 
'''

def point_click(filepath):

	img = cv2.imread(filepath,-1)
	_, ax = plt.subplots(figsize=(9, 9))
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	imgplot = ax.imshow(img)
	x_c, y_c = plt.ginput(1, mouse_stop=2)[0]		#y_c is the row while x_c is the column
	print(y_c,x_c)
	cv2.destroyAllWindows()

	return y_c,x_c

def find_detection_rate(error_array,distance_metric):
	correct_detections = len(error_array[np.where( error_array < distance_metric)])
	return correct_detections/error_array.shape[0]

if __name__ == "__main__":

	images_path = sys.argv[1]
	images_path = '../image_data/'	#Remove this
	centroid_using_algo_csv = sys.argv[2]
	centroid_using_algorithm = np.genfromtxt(centroid_using_algo_csv, delimiter=',')
	distance_metric = 20 #(in pixels)


	for i,filename in enumerate(os.listdir(images_path)):

		y_pc,x_pc = point_click(images_path + filename)

		if(i==0):
			centroid_using_point_click = np.array([y_pc,x_pc])
		else:
			centroid_using_point_click = np.vstack((centroid_using_point_click,np.array([y_pc,x_pc])))

	error_array = np.linalg.norm(np.abs(centroid_using_point_click - centroid_using_algorithm),axis = 1)
	detection_rate = find_detection_rate(error_array,distance_metric)
	print("Detection rate is ",detection_rate*100," % \n")





