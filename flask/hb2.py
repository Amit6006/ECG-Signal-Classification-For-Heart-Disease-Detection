# import cv2
# import numpy as np
# from scipy.signal import find_peaks

# # Load the image
# # image = cv2.imread('scologram.png')
# image = cv2.imread(r'C:\Users\Harsh\OneDrive\Desktop\Sleep Pattern Recog\cnn\ecgdataset\arr\1.jpg')


# # Convert to grayscale
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # Apply Gaussian blur to reduce noise
# blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# # Perform Canny edge detection
# edges = cv2.Canny(blurred_image, threshold1=30, threshold2=100)

# # Display the edges
# cv2.imshow('Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Convert edge image to 1D signal by summing along the y-axis
# signal = np.sum(edges, axis=0)

# # Normalize the signal
# signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# # Detect peaks in the signal
# peaks, _ = find_peaks(signal, height=0.5, distance=50)  # Adjust height and distance as needed

# # Print number of detected peaks
# print(f'Detected R-peaks: {len(peaks)}')

# # Calculate heart rate
# time_interval_seconds = 10  # Assuming the x-axis of the image represents 10 seconds
# heart_rate = (len(peaks) / time_interval_seconds) * 60

# print(f'Heart Rate: {heart_rate} bpm')


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load the image
# image = cv2.imread('scologram.png')
image = cv2.imread(r'C:\Users\Harsh\OneDrive\Desktop\Sleep Pattern Recog\cnn\ecgdataset\chf\4.jpg')


# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform Canny edge detection
edges = cv2.Canny(gray_image, threshold1=50, threshold2=150)

# Display the edges
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Convert edge image to 1D signal by summing along the y-axis
signal = np.sum(edges, axis=0)

# Normalize the signal
signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))

# Detect peaks in the signal
peaks, properties = find_peaks(signal, height=0.5, distance=30)  # Adjust height and distance as needed

# Print number of detected peaks
print(f'Detected R-peaks: {len(peaks)}')

# Calculate heart rate
time_interval_seconds = 4  # Assuming the x-axis of the image represents 10 seconds
heart_rate = (len(peaks) / time_interval_seconds) * 60

print(f'Heart Rate: {round(heart_rate)} bpm')

# Plot the signal and detected peaks
plt.figure(figsize=(10, 4))
plt.plot(signal, label='ECG Signal')
plt.plot(peaks, signal[peaks], 'rx', label='Detected Peaks')
plt.title('Detected Peaks in ECG Signal')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()
