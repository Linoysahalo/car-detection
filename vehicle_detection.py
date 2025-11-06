import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"
os.environ["TF_ENABLE_ONEDNN_OPTS"]="0"
import cvlib as cv

# function to detect vehicles in the image, filter out small or edge-cut vehicles
# and return a list of filtered vehicles object with their bounding boxes, area, area compared to original image, and aspect ratios
def detect_vehicles(image):
    # detecting objects in the image using YOLOv4-tiny model
    bounding_boxes,labels,confidences=cv.detect_common_objects(
        image,confidence=0.3,model='yolov4-tiny'
    )
    # extracting image dimensions and area
    img_h,img_w=image.shape[:2]
    img_area=img_w*img_h
    # margins defining to filter out edge-cut vehicles (only the objects that really close to the edge will be filtered out)
    margin_x=int(img_w*0.01)
    margin_y=int(img_h*0.01)

    cx_img,cy_img=img_w/2.0,img_h/2.0

    detected_vehicles=[]

    # filtering vehicles from detected objects
    for box,object_label,confidence in zip(bounding_boxes,labels,confidences):
        if object_label in ['car','bus','truck','motorcycle']:
            x1,y1,x2,y2=box

            # vehicle object fields calculation
            width,height=x2-x1,y2-y1
            box_area=width*height
            area_from_image=box_area/float(img_area)
            aspect_ratio=width/float(height)

            # edge-cut vehicles (only filter if also small and off-center)
            cx,cy=x1+width/2.0,y1+height/2.0
            touches_edge=(x1<=margin_x or y1<=margin_y or x2>=img_w-margin_x or y2>=img_h-margin_y)
            smallish=(area_from_image<0.05 or width<0.15*img_w or height<0.15*img_h)
            off_center=(abs(cx-cx_img)>0.38*img_w or abs(cy-cy_img)>0.38*img_h)
            if touches_edge and smallish and off_center:
                continue

            detected_vehicles.append({
                "box":(x1,y1,x2,y2),
                "area":box_area,
                "area_from_image":area_from_image,
                "aspect_ratio":aspect_ratio
            })

    # finding the biggest vehicle by area
    if detected_vehicles:
        max_area=max(v["area"] for v in detected_vehicles)

        # keeping only vehicles that are at least 1/4 of the biggest one
        detected_vehicles=[v for v in detected_vehicles if v["area"]>=max_area/4]

        # sorting vehicles by area 
        detected_vehicles.sort(key=lambda v:v["area"],reverse=True)

    return detected_vehicles

# function to classify vehicle based on its ratio
# receives vehicle object and returns the vehicle type (string)
def classify_vehicle(vehicle_data):
    aspect_ratio=vehicle_data["aspect_ratio"]

    # narrow and tall- truck
    # almost square- bus
    # little wider than a square- private car
    # wider- not straight forward

    if aspect_ratio<=0.8:
        return "Truck"
    elif 0.8<aspect_ratio<=1:
        return "Bus"
    elif 1<aspect_ratio<=1.6:
        return "car"
    else:
        return "cannot classify images of vehicles that are facing sideways or were taken from an angle"
