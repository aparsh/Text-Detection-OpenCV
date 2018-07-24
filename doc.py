import cv2
import numpy as np
import os
import face_detect


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def show_wait_destroy(winname, img):
    cv2.imshow(winname, img)
    cv2.moveWindow(winname, 500, 0)
    cv2.waitKey(0)
    cv2.destroyWindow(winname)


if __name__ == '__main__':

    # Read image
    image_path = ""
    large = cv2.imread(image_path)



    # Grayscale
    try:
        gray = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)
    except:
        gray = cv2.cvtColor(large, cv2.COLOR_RGB2GRAY)

    # Get location of face
    face = face_detect.get_face(large, gray)

    gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray,3)

    # Canny
    edges = cv2.Canny(np.asarray(gray), 100, 200)

    # Dilation #1
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilate1 = cv2.dilate(edges,morph_kernel,iterations = 4)

    # Erosion #1
    erode_kernel = np.ones((5,1),np.uint8)
    erosion1 = cv2.erode(dilate1,erode_kernel,iterations = 1)


    # Dilation #2
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate2 = cv2.dilate(erosion1,morph_kernel,iterations = 4)


    # Erosion #2
    erode_kernel = np.ones((3,1),np.uint8)
    erosion2 = cv2.erode(dilate2,erode_kernel,iterations = 2)


    # Dilation #3
    # morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # dilate3 = cv2.dilate(erosion2,morph_kernel,iterations = 3)
    #

    # Closing to remove small holes #1
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # connect horizontally oriented regions
    closing1 = cv2.morphologyEx(erosion2, cv2.MORPH_CLOSE, morph_kernel)

    # Closing #2
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # connect horizontally oriented regions
    closing2 = cv2.morphologyEx(closing1, cv2.MORPH_CLOSE, morph_kernel)

    erode_kernel = np.ones((3,1),np.uint8)
    erosion3 = cv2.erode(closing2,erode_kernel,iterations = 3)

    # Dilation to group together horizontal words
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    dilate4 = cv2.dilate(closing2,morph_kernel,iterations = 3)

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    # connect horizontally oriented regions
    closing3= cv2.morphologyEx(dilate4, cv2.MORPH_CLOSE, morph_kernel)


    # cv2.imwrite("das.jpg",closing3)

    # Contours
    im2, contours, hierarchy = cv2.findContours(closing2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    useful = []
    for contour in contours:
        x, y, rect_width, rect_height = cv2.boundingRect(contour)
        ratio = rect_width/ rect_height
        area = rect_height * rect_width

        # Remove unwanted boxes with reference to face
        if x < face[0] + face[2] or y<face[1]- 0.8*(face[3]):
            continue

        # Remove small boxes by area with reference to face
        if area < 7500:
            continue
        useful.append(contour)
        # Draw rectangles
        large = cv2.rectangle(large, (x, y), (x + rect_width, y+ rect_height), (0, 255, 0), 3)

    # Contours sorted top to bottom
    sorted_cnts,boxes = sort_contours(useful,"top-to-bottom")
    print(len(sorted_cnts))
    print("Countours sorted")

    # Create dictionary containing each key as level of y-coordinate
    i = 0
    levels = {}
    contour_levels = {}
    levels[boxes[i][1]] = [boxes[i]]
    contour_levels[boxes[i][1]] = [sorted_cnts[i]]
    current = boxes[i][1]
    i = i+1
    y_thresh = 50
    x_thresh = 70
    while i in range(len(boxes)):
        if abs(current-boxes[i][1]) <= y_thresh:
            levels[current].append(boxes[i])
            contour_levels[current].append(sorted_cnts[i])

        else:
            current = boxes[i][1]
            levels[current] = [boxes[i]]
            contour_levels[current] = [sorted_cnts[i]]
        i += 1

    # Sort the dictionary by x-coordinates
    for key in contour_levels.keys():
        sorted_horz,boxes = sort_contours(contour_levels[key],"left-to-right")
        contour_levels[key] = sorted_horz
        levels[key] = boxes

    # Group together boxes that are on the same line (similar y-coordinates) according to the x_thresh
    group_boxes = {}
    for key in levels.keys():
        i = 0
        horz_expanse = levels[key][i][0] + levels[key][i][2]
        current = key
        group_boxes[current] = [levels[key][i]]
        i += 1
        while i in range(len(levels[key])):
            if abs(horz_expanse - levels[key][i][0]) <= x_thresh:
                horz_expanse = levels[key][i][0] + levels[key][i][2]
                group_boxes[current].append(levels[key][i])
            else:
                current = levels[key][i][0]
                group_boxes[current] = [levels[key][i]]
                horz_expanse = levels[key][i][0] + levels[key][i][2]
            i += 1

    # Plot the rectangles on the image
    outputs = []
    for key in group_boxes.keys():
        x_s = []
        x_max_s = []
        y_s = []
        y_max_s = []
        for x in group_boxes[key]:
            x_s.append(x[0])
            x_max_s.append(x[0]+x[2])
            y_s.append(x[1])
            y_max_s.append(x[1]+x[3])
        large = cv2.rectangle(large, (min(x_s), min(y_s)), (max(x_max_s), max(y_max_s)), (0, 255, 0), 3)

        buffer = 1
        arr = large[min(y_s)-buffer:max(y_max_s)+buffer,min(x_s)-buffer:max(x_max_s)+buffer]


    cv2.imwrite("result.jpg",large)