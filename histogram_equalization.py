import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

#Directory of the images to be equalized
path = os.getcwd()+"/adaptive_hist_data" 
# print(path)

#Directory where we save the equalized images
saving_path = os.getcwd()+"/Histogram_Equalization"

def compute_cdf(bins):
    """
    Parameters
    ----------
    bins : Bins
        Frequency of the intensity values (256 bins).

    Returns
    -------
    cdf : list
        Cumulative distribution function for the bins.

    """
    cdf = []
    N = sum(bins) #Total Frequency of the intensity
    
    # print("N: ",N)
    for h in range(0,len(bins)):
        ci = bins[:h+1]
        cdf.append(sum(ci)/N)

    return cdf

try:
    for image in os.listdir(path):
        frame = cv2.imread(path+"/"+image) #Reading the image
        # print(frame.shape)
        # cv2.imshow("Frame",frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        #BLUE CHANNEL
        b_frame = frame[:,:,0]
        b_bins = []
        #Generating the Bins (256) [for 0-255 intensity levels]
        for i in range(0,256):
            b_bins.append(len(np.where(b_frame==i)[0]))
        
        # Computing the CDF for the blue channel    
        cdf = compute_cdf(b_bins)
        
        # fig2,ax2 = plt.subplots()
        # ax2.plot(cdf)
        
        #Replacing the pixel values with the computed CDF
        for height in range(0,frame.shape[0]):
            for width in range(0,frame.shape[1]):
                intensity = b_frame[height][width]
                cdf_mult  = cdf[intensity]
                b_frame[height][width] = np.uint8(cdf_mult*255)
        
        #GREEN CHANNEL
        g_frame = frame[:,:,1]
        g_bins = []
        #Generating the bins
        for i in range(0,256):
            g_bins.append(len(np.where(g_frame==i)[0]))
        
        # Computing the CDF
        cdf = compute_cdf(g_bins)
        # fig2,ax2 = plt.subplots()
        # ax2.plot(cdf)
        
        #Replacing the pixel values with the computed CDF
        for height in range(0,frame.shape[0]):
            for width in range(0,frame.shape[1]):
                intensity = g_frame[height][width]
                cdf_mult  = cdf[intensity]
                g_frame[height][width] = np.uint8(cdf_mult*255)
        
        #RED CHANNEL
        r_frame = frame[:,:,2]
        r_bins = []
        #Generating the Bins
        for i in range(0,256):
            r_bins.append(len(np.where(r_frame==i)[0]))
        
        #Computing the CDF
        cdf = compute_cdf(r_bins)
        # fig2,ax2 = plt.subplots()
        # ax2.plot(cdf)
        # print(len(cdf))
        
        #Replacing the pixel values with the computed CDF
        for height in range(0,frame.shape[0]):
            for width in range(0,frame.shape[1]):
                intensity = r_frame[height][width]
                cdf_mult  = cdf[intensity]
                r_frame[height][width] = np.uint8(cdf_mult*255)
        
        # Merging the individual BGR channels to get the single image
        
        req_frame = cv2.merge((b_frame,g_frame,r_frame))
        # fig,axes = plt.subplots(3)
        # axes[0].plot(b_bins)
        # axes[1].plot(g_bins)
        # axes[2].plot(r_bins)
        
        # plt.show()
        # cv2.imshow("Frame",req_frame)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(saving_path+"/"+image)
        cv2.imwrite(saving_path+"/"+image,req_frame) #Saving the equalized images
except:
    print("Couldn't open image from the path.")