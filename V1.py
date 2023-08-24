import io
import cv2
import numpy as np
#import pytesseract
from PIL import Image as P_Image

def CannyThreshold(Captured_Image):  
    Gray_Scale = cv2.cvtColor(Captured_Image, cv2.COLOR_BGR2GRAY)
    bilateral_Filter = cv2.bilateralFilter(Gray_Scale, 11, 17, 17)

    Image_Median = np.median(Captured_Image)
    Lower_Threshold = max(0, (0.7 * Image_Median))
    Upper_Threshold = min(255, (0.7 * Image_Median)) 

    CannyEdged = cv2.Canny(bilateral_Filter,Lower_Threshold,Upper_Threshold)
    
    return CannyEdged


def Cal_Density(Cap_Image,Ref_Image):
    Ref_indices = np.where(Ref_Image != [0])
    Cap_indices = np.where(Cap_Image != [0])
    Ref_count = len(Ref_indices[0])
    Cap_count = len(Cap_indices[0])
    percent = ((Ref_count )/ Cap_count) * 100
    if(percent>=100):
        return 0
    return percent





Ref_Imagepath = "./Images/ref-image.jpg"
Lane1_Imagepath = "./Images/lane1.jpg"
Lane2_Imagepath = "./Images/lane2.jpg"
Lane3_Imagepath = "./Images/lane3.jpg"
Lane4_Imagepath = "./Images/lane4.jpg"

Ref_Image = cv2.imread(Ref_Imagepath)
Lane1_Image = cv2.imread(Lane1_Imagepath)
Lane2_Image = cv2.imread(Lane2_Imagepath)
Lane3_Image = cv2.imread(Lane3_Imagepath)
Lane4_Image = cv2.imread(Lane4_Imagepath)

Canny_Ref = CannyThreshold(Ref_Image)
Canny_1 = CannyThreshold(Lane1_Image)
Density_1 = Cal_Density(Canny_1, Canny_Ref)
Time_allocation_1 = 5 * round((Density_1 / 20))
print(Density_1, Time_allocation_1)
Canny_2 = CannyThreshold(Lane2_Image)
Density_2 = Cal_Density(Canny_2, Canny_Ref) 
Time_allocation_2 = 5 * round((Density_2 / 20))
print(Density_2, Time_allocation_2)
Canny_3 = CannyThreshold(Lane3_Image)
Density_3 = Cal_Density(Canny_3, Canny_Ref) 
Time_allocation_3 = 5 * round((Density_3 / 20))
print(Density_3, Time_allocation_3)
Canny_4 = CannyThreshold(Lane4_Image)
Density_4 = Cal_Density(Canny_4, Canny_Ref) 
Time_allocation_4 = 5 * round((Density_4 / 20))
print(Density_4, Time_allocation_4)






