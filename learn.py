import os
import os.path
import cv2
import glob
import imutils


CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


# Get a list of all the captcha images we need to process
captcha_image_files = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

# loop over the image paths
# Since the filename contains the captcha text (i.e. "2A2X.png" has the text "2A2X"),
captcha_image_file = captcha_image_files[2]
filename = os.path.basename(captcha_image_file)
captcha_correct_text = os.path.splitext(filename)[0]

# Load the image and convert it to grayscale
image = cv2.imread(captcha_image_file)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("Output", image)
# cv2.waitKey()
# cv2.imshow("Output", gray)
# cv2.waitKey()

# Add some extra padding around the image
gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
# print("Padded")
# cv2.imshow("Output", gray)
# cv2.waitKey()

# threshold the image (convert it to pure black and white)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
# print('Threshold')
# cv2.imshow("thresh", thresh)
# cv2.waitKey()

# find the contours (continuous blobs of pixels) the image
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
letter_image_regions = []
for contour in contours:
  # Get the rectangle that contains the contour
  (x, y, w, h) = cv2.boundingRect(contour)

  # Compare the width and height of the contour to detect letters that
  # are conjoined into one chunk
  if w / h > 1.25:
      # This contour is too wide to be a single letter!
      # Split it in half into two letter regions!
      half_width = int(w / 2)
      letter_image_regions.append((x, y, half_width, h))
      letter_image_regions.append((x + half_width, y, half_width, h))
  else:
      # This is a normal letter by itself
      letter_image_regions.append((x, y, w, h))

# Sort the detected letter images based on the x coordinate to make sure
# we are processing them from left-to-right so we match the right image
# with the right letter
letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

for letter_bounding_box, letter_text in zip(letter_image_regions, captcha_correct_text):
  # Grab the coordinates of the letter in the image
  x, y, w, h = letter_bounding_box

  # Extract the letter from the original image with a 2-pixel margin around the edge
  letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]
  cv2.imshow('t', letter_image)
  cv2.waitKey()
