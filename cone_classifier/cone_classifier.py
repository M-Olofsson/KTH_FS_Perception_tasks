import cv2
import numpy as np
import matplotlib.pyplot as plt

"""File path to the directory with cone images"""
file_path = r'//home/mattias/KTH_FS_Perception_tasks/cone_classifier/resources/'

class Cone:
    """Cone objects with img as numpy array, and color string label"""
    def __init__(self, img, color):
        self.img = img
        self.color = color

def main():
    blue_cones, yellow_cones = read_imgs()
    correct_blue_count = classify_cones(blue_cones)
    correct_yellow_count = classify_cones(yellow_cones)
    color = np.array(['Blue','Yellow'])
    percentage = np.array([(correct_blue_count/len(blue_cones))*100,(correct_yellow_count/len(yellow_cones))*100])
    plt.bar(color,percentage)
    plt.show()


def read_imgs():
    blue_cones = []
    yellow_cones = []

    for i in range(15):
        file_name = file_path+'blue_'+str(i)+'.png'
        img = cv2.imread(file_name)
        cone = Cone(img,'blue')
        blue_cones.append(cone)

    for i in range(14):
        file_name = file_path+'yellow_'+str(i)+'.png'
        img = cv2.imread(file_name)
        cone = Cone(img,'yellow')
        yellow_cones.append(cone)

    return blue_cones, yellow_cones


def classify_cones(cones):
    """Takes in list of cones with same colour, classifies their color, and returns
     number of correct classifications"""

    correct_count = 0

    """Create mask boundaries"""
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([130, 255, 255])
    lower_yellow = np.array([20, 150, 50])
    upper_yellow = np.array([40, 255, 255])

    """Classify each cone"""
    for cone in cones:
        img_hsv = cv2.cvtColor(cone.img, cv2.COLOR_BGR2HSV)
        blue_mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
        yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)
        if np.sum(blue_mask) > np.sum(yellow_mask):
            classification = 'blue'
        else:
            classification = 'yellow'
        if cone.color == classification:
            correct_count += 1
    return correct_count

if __name__ == '__main__':
    main()