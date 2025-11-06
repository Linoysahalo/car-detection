from math import sqrt
import cv2, re, easyocr
import numpy as np
import easyocr
from tkinter import END
from vehicle_detection import detect_vehicles, classify_vehicle
import random
from PIL import Image, ImageTk

reader=easyocr.Reader(['en'], gpu=False) 

#####---COLOR---#####    
# function to classify vehicle color 
# receives vehicle data and ROI and returns its color (string)
def classify_color(vehicle_data,image):
    x1,y1,x2,y2=vehicle_data["box"]
    h=y2-y1
    w=x2-x1

    # now we selecting a ROI in the middle of the car, so its less likely to have windows and
    # things that not related to the car pure color
    
    # taking of the side margins
    mx=int(0.12*w)
    x1_new=max(x1,x1+mx)
    x2_new=min(x2,x2-mx)

    # if vehicle type is Bus -> use bottom third of the ROI for color
    if vehicle_data.get("__type")=="Bus":
        yt=y1+int(0.67*h)
        yb=y2
    else:
        # centering the box
        yt=y1+int(0.33*h)
        yb=y1+int(0.55*h)

    yt=max(y1,min(yt,y2-2))
    yb=max(yt+2,min(yb,y2))

    roi=image[yt:yb,x1_new:x2_new]
    if roi.size==0:
        return "Unknown"
    
    # deciding if the car is colored or not
    # value and saturation of the ROI
    hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    S=hsv[:,:,1].astype(np.float32)
    V=hsv[:,:,2].astype(np.float32)
    lowS_ratio=float(np.mean(S<50)) # what is the low saturation ratio

    # difference and spread of the color
    B,G,R=roi[:,:,0],roi[:,:,1],roi[:,:,2]
    spread=(np.maximum.reduce([R,G,B])-np.minimum.reduce([R,G,B])).astype(np.float32)
    lowDiff_ratio=float(np.mean(spread<28))

    # precentage of value (10%, 50%, 90%)
    V_p10,V_p50,V_p90=np.percentile(V,[10,50,90]).tolist()

    # if the color is faded or low value, its probably black/white/grey
    is_achro=(
        lowDiff_ratio>=0.60 or
        lowS_ratio>=0.60 or
        V_p90<=135 or
        V_p10>=195
    )

    if is_achro:
        return black_white_grey_identify(roi)
    return colored_car_identify(roi)

# function to decide if the color is black/white/grey
# receives vehicle ROI and returns color (string)
def black_white_grey_identify(roi):
    # gray scale
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY).astype(np.float32)
    if min(gray.shape[:2])>=6:
        gray=cv2.GaussianBlur(gray,(3,3),0) # soft blur

    # taking precentages
    p10,p50,p75,p90=np.percentile(gray,[10,50,75,90]).tolist()
    bright_ratio_raw=float(np.mean(gray>220)) # how many pixels are bright
    dark_ratio_raw=float(np.mean(gray<60)) # how many are dark

    # stretch if we need to deal with shadows on white 
    lo=float(np.percentile(gray,5))
    hi=float(np.percentile(gray,98))
    if hi-lo<5:
        gray_st=gray.copy()
    else:
        gray_st=(gray-lo)*(255.0/(hi-lo))
        gray_st=np.clip(gray_st,0,255)

    # precentages (for grey this time)
    p50n,p75n,p90n=np.percentile(gray_st,[50,75,90]).tolist()
    bright_ratio_norm=float(np.mean(gray_st>210)) # bright ratio

    # white check
    if (p75>=200) or (p90>=225) or (bright_ratio_raw>=0.25 and p50>=170) \
       or (p75n>=210) or (p50n>=185) or (bright_ratio_norm>=0.40):
        return "White"

    # black check
    if (p90<=120) or (p50<=105) or (dark_ratio_raw>=0.60):
        return "Black"

    return "Grey"

# function to calculate the avergae RGB color and return the closest color
# receives car ROI and returns the color (string)
def colored_car_identify(roi):
    # dinding the average BGR and converting to RGB
    avg_bgr=np.mean(roi.reshape(-1,3),axis=0)
    avg_rgb=(avg_bgr[2],avg_bgr[1],avg_bgr[0])  

    # defining the boundries to every color
    reference_colors={
        "brown":(115,75,45),   
        "red":(210,35,35),   
        "blue":(60,100,160),   
        "sky blue":(170,255,250),   
        "green":(60,170,60),   
        "yellow":(235,210,30),  
    }

    # finding the similarity to every color using rgb_similarity and choose the best similar
    sims={name:rgb_similarity(avg_rgb,ref) for name,ref in reference_colors.items()}
    return max(sims,key=sims.get)

# function to calculate the similarity of the average RGB color to every color by the pearson poison
# receives R/G/B tuples and retuens value
def rgb_similarity(c1,c2):
    s1,s2=sum(c1),sum(c2)
    sp1,sp2=sum([x**2 for x in c1]),sum([x**2 for x in c2])
    sp=sum([a*b for a,b in zip(c1,c2)])
    try:
        return (sp-(s1*s2/3.0))/sqrt((sp1-(s1**2)/3.0)*(sp2-(s2**2)/3.0)) # pearson
    except ZeroDivisionError:
        return 0

#####---LICENSE PLATE----#####   
# function to format the final plate number to israeli
# receives numbers string and returns formatted string
def _format_israeli_plate(digits:str)->str:
    if len(digits)==8: # 123-45-678
        return f"{digits[:3]}-{digits[3:5]}-{digits[5:]}"
    if len(digits)==7: # 12-345-67
        return f"{digits[:2]}-{digits[2:5]}-{digits[5:]}"
    return digits

# function to make final clean the plate after the filters- it filters out small and
# not digit-sized things
# receives and returns plate
def clean_specks_and_black_bg(bin_img):
    H,W=bin_img.shape
    bin_img=(bin_img>0).astype(np.uint8)*255
    k_open=cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    opened=cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,k_open,iterations=1)

    num,labels,stats,_=cv2.connectedComponentsWithStats(opened,connectivity=8)
    min_h=max(8,int(0.30*H))              
    min_area=max(30,int(0.0015*H*W))
    cleaned=np.zeros_like(opened)
    for i in range(1,num):
        x,y,w,h,area=stats[i]
        if h>=min_h and area>=min_area:
            cleaned[labels==i]=255

    k_close=cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))
    cleaned=cv2.morphologyEx(cleaned,cv2.MORPH_CLOSE,k_close,iterations=1)
    return cleaned

# function to find the digits only area in the plate
def filter_and_crop_digits(bw_inv):
    H,W=bw_inv.shape
    bw_clean=cv2.morphologyEx(bw_inv,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_RECT,(3,1)),iterations=1)
    proj=np.sum(bw_clean>0,axis=1).astype(np.float32)/max(W,1)
    win_h=max(int(0.75*H),12)
    cumsum=np.cumsum(np.pad(proj,(1,0)))
    scores=cumsum[win_h:]-cumsum[:-win_h]
    y0=int(np.argmax(scores))
    y1=min(H,y0+win_h)

    pad_y=max(2,int(0.20*H))
    extra=int(0.08*H)
    y0=max(0,y0-pad_y-extra)
    y1=min(H,y1+pad_y+extra)

    band=bw_inv[y0:y1,:]
    contours,_=cv2.findContours(band,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    mask_band=np.zeros_like(band)
    keep_boxes=[]
    min_h=int(0.30*(y1-y0))
    min_w_lo=int(0.03*W)
    min_w_hi=int(0.25*W)

    for c in contours:
        x,y,w,h=cv2.boundingRect(c)
        if not (h>=min_h or (min_w_lo<=w<=min_w_hi)):
            continue
        roi=band[y:y+h,x:x+w]
        white_ratio=float(np.mean(roi>0))
        if not (0.03<=white_ratio<=0.90):
            continue
        keep_boxes.append((x,y,w,h))
        cv2.drawContours(mask_band,[c],-1,255,thickness=cv2.FILLED)

    filtered_band=cv2.bitwise_and(band,mask_band)
    filtered_full=np.zeros_like(bw_inv)
    filtered_full[y0:y1,:]=filtered_band

    if not keep_boxes:
        digits_strip=filtered_full[y0:y1,:]
    else:
        xs=[x for (x,_,w,_) in keep_boxes]
        xe=[x+w for (x,_,w,_) in keep_boxes]
        x_min=max(0,min(xs))
        x_max=min(W,max(xe))
        pad_x=int(0.06*W)
        x0=max(0,x_min-pad_x)
        x1=min(W,x_max+pad_x)
        digits_strip=filtered_full[y0:y1,x0:x1]

    h,w=digits_strip.shape[:2]
    left_cut=min(int(0.10*w),max(w-1,0))
    bottom_keep=max(int(round((1.0-0.10)*h)),1)
    digits_strip=clean_specks_and_black_bg(digits_strip)
    digits_strip=digits_strip[:bottom_keep,left_cut:]
    return filtered_full,mask_band,digits_strip

# function to run OCR to find the digits, and return the string if its length is 7/8
# receives regions and returns formatted string
def ocr_on_regions(regions,roi_rs):
    best_text=""
    for (x,y,cw,ch) in regions:
        crop=roi_rs[y:y+ch,x:x+cw]
        g=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        bw_inv=cv2.threshold(g,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        _,_,digits_strip=filter_and_crop_digits(bw_inv)
        prep_rgb=cv2.cvtColor(digits_strip,cv2.COLOR_GRAY2RGB)
        results=reader.readtext(prep_rgb,detail=0,paragraph=False,allowlist="0123456789")
        for t in results:
            digits=re.sub(r"\D","",t)
            if len(digits) in (7,8):
                return _format_israeli_plate(digits)
    return best_text

# function to detect license plate from car ROI and return formatted plate number
def detect_license_plate(roi):
    if roi is None or roi.size==0:
        return ""
    try:
        h,w=roi.shape[:2]
        scale=600.0/max(h,w)
        roi_rs=cv2.resize(roi,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC) if scale<1.5 else roi.copy()

        gray=cv2.cvtColor(roi_rs,cv2.COLOR_BGR2GRAY)
        gray=cv2.bilateralFilter(gray,9,35,35)
        rect=cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
        blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,rect)
        gradX=cv2.Sobel(blackhat,ddepth=cv2.CV_32F,dx=1,dy=0,ksize=-1)
        gradX=np.absolute(gradX)
        gradX=(255*((gradX-gradX.min())/(gradX.max()-gradX.min()+1e-6))).astype("uint8")
        gradX=cv2.morphologyEx(gradX,cv2.MORPH_CLOSE,rect,iterations=2)
        _,thresh=cv2.threshold(gradX,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        thresh=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel,iterations=1)
        thresh=cv2.dilate(thresh,kernel,iterations=1)
        contours,_=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        candidates=[]
        for c in contours:
            x,y,cw,ch=cv2.boundingRect(c)
            if cw*ch<0.002*(roi_rs.shape[0]*roi_rs.shape[1]): 
                continue
            ar=cw/float(max(ch,1))
            if 2.0<=ar<=6.5:
                candidates.append((x,y,cw,ch))

        if candidates:
            candidates=sorted(candidates,key=lambda b:b[2],reverse=True)[:5]
            text=ocr_on_regions(candidates,roi_rs)
            if text:
                return text

        roi_rgb=cv2.cvtColor(roi_rs,cv2.COLOR_BGR2RGB)
        results=reader.readtext(roi_rgb,detail=0,paragraph=False,allowlist="0123456789")
        for t in results:
            digits=re.sub(r"\D","",t)
            if len(digits) in (7,8):
                return _format_israeli_plate(digits)
        return ""
    except Exception:
        return ""

# function to process the image, detect and classify vehicles, and update the GUI
def process_image(path,output_text,image_label):
    try:
        image=cv2.imread(path)
        if image is None:
            output_text.insert(END,f"failed to read picture from: {path}\n")
            return

        vehicles=detect_vehicles(image)
        output_image=image.copy()

        if vehicles:
            colors=[(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for _ in vehicles]

            for vehicle_id,(v,color) in enumerate(zip(vehicles,colors),start=1):
                x1,y1,x2,y2=v["box"]

                cv2.rectangle(output_image,(x1,y1),(x2,y2),color,4)
                label=f"#{vehicle_id}"
                font=cv2.FONT_HERSHEY_SIMPLEX
                scale=0.9
                thickness=2
                (tw,th),_=cv2.getTextSize(label,font,scale,thickness)
                cv2.rectangle(output_image,(x1,y1-th-10),(x1+tw,y1),color,-1)
                cv2.putText(output_image,label,(x1,y1-5),font,scale,(255,255,255),thickness)

                # classify type first
                vehicle_type=classify_vehicle(v)
                # pass the type to color logic via the vehicle dict
                v["__type"]=vehicle_type

                vehicle_color=classify_color(v,image)

                vehicle_roi=image[y1:y2,x1:x2]
                plate_text=detect_license_plate(vehicle_roi)
                if not plate_text:
                    # expand ROI and retry
                    H,W=image.shape[:2]
                    pad_x=int(0.08*(x2-x1))
                    pad_y=int(0.12*(y2-y1))
                    xe1=max(0,x1-pad_x)
                    ye1=max(0,y1-pad_y)
                    xe2=min(W,x2+pad_x)
                    ye2=min(H,y2+pad_y)
                    roi_expanded=image[ye1:ye2,xe1:xe2]
                    plate_text=detect_license_plate(roi_expanded)

                output_text.insert(
                    END,
                    f"#{vehicle_id}: {vehicle_type}, color: {vehicle_color}, License Plate: {plate_text if plate_text else 'N/A'}\n"
                )
        else:
            output_text.insert(END,"sorry, no vehicles detected\n")

        image_rgb=cv2.cvtColor(output_image,cv2.COLOR_BGR2RGB)
        image_pil=Image.fromarray(image_rgb).resize((300,300))
        imgtk=ImageTk.PhotoImage(image=image_pil)
        image_label.config(image=imgtk)
        image_label.image=imgtk

    except Exception as e:
        output_text.insert(END,f"error {e}\n")
