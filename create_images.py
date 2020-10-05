from PIL import Image, ImageEnhance, ImageOps
import PIL
import random
from math import cos,sin
import shutil

from os import listdir
from os.path import isfile, join
import glob, os
import numpy as np
import cv2
import cv2.aruco as aruco
from time import sleep

from math import sqrt


class Coords: 
    def __init__(self, x, y, img_width,img_height): 
        self.x = x 
        self.y = y 
        self.img_width = img_width
        self.img_height = img_height
    #end

    def overlap(self,coords):
        if(self.x >= coords.x+coords.img_width or coords.x >= self.x+self.img_width):
            return False
        #end

        if(self.y >= coords.y+coords.img_height or coords.y >= self.y+self.img_height):
            return False
        #end

        return True
    #end
    
#end

def load_images(background_size,signs_size):

    #Signs to Detect
    detect_signs = ["0-1.png","0-2.png","1-2.png","2-2.png","0-3.png","1-3.png","2-3.png","0-4.png","1-4.png","2-4.png"]
    real_signs = {}
    for i,sign in enumerate(detect_signs):
        real_signs[detect_signs[i][0:3]] = Image.open('images/'+sign).convert('RGBA')
    #end

    #Fake signs
    fakes = ["sign1.png","sign3.png","sign5.png","sign6.png"]
    fake_signs = {}
    for i,sign in enumerate(fakes):
        fake_signs["Fake"+str(i)] = Image.open('images/'+sign).resize(signs_size).convert('RGBA')
    #end



    #Load backgrounds
    backgrounds = [f for f in listdir('backgrounds/') if isfile(join('backgrounds/', f))]
    output_backgrounds = []
    for background in backgrounds:
        output_backgrounds.append(Image.open('backgrounds/'+background).resize(background_size).convert('RGBA'))
    #end

    return output_backgrounds, real_signs, fake_signs
#end

def section(img,sq_size):
    #Split the image into a square region with random cooridinates.
    #So the background is not identical in each picture
    # sq_size = 1500
    width,height = img.size
    
    x_pos = random.randint(0,width-sq_size)
    y_pos = x_pos + sq_size

    crop_pos = (x_pos,x_pos,y_pos,y_pos)
    return img.crop(crop_pos)
#end

def text_for_creation(f,img_num,width,height):
    temp = """<annotation>
  <folder>VOC2007</folder>
  <filename>{0}.jpg</filename>
  <source>
    <database>Self-made database</database>
  </source>
  <size>
    <width>{1}</width>
    <height>{2}</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>"""

    f.write(temp.format(str(img_num),str(width),str(height)))
#end


def save_to_text_file(f,label,x_pos,y_pos,width,height):
    #Save the super imposed images coordinates into a text file that yolo can use.
    
    labels = {"0":"Aruco",
                "1":"Dangerous_Goods",
                "2":"Chemical_Sign"}

    write_text = """\n  <object>
    <name>{0}</name>
    <pose>Unspecified</pose>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox>
      <xmin>{1}</xmin>
      <ymin>{2}</ymin>
      <xmax>{3}</xmax>
      <ymax>{4}</ymax>
    </bndbox>
  </object>"""

    f.write(write_text.format(labels.get(str(label)), str(x_pos),str(y_pos),str(round(x_pos+width)),str(round(y_pos+height))))
#end


def check_overlap(img_width,img_height,back_width,back_height, previous_pasted_coords):
    #Ensure image coordinates do not overlap with a previous image
    #Get a random postion to place image
    x_pos = random.randint(0,back_width-img_width)
    y_pos = random.randint(0,back_height-img_height)

    x1 = (x_pos+(img_width/2))/back_width
    y1 = (y_pos+(img_height/2))/back_width
    
    temp = Coords(x1,y1,img_width,img_height)
    #Loop through all previously pasted images
    for pasted in previous_pasted_coords:
        if temp.overlap(pasted):
            try:
                return check_overlap(img_width,img_height,back_width,back_height,previous_pasted_coords)
            except:
                return -100,-100
            #end
        #end
    #end

    #Image does not overlap with any other image, so return coords
    return x_pos, y_pos
#end

def noisy(image):
    #Add Gausian noise to image.
    row,col= image.size
    ch = 4
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image + gauss
    return Image.fromarray(np.uint8(noisy))
#end

def rotate(img):
    return img.rotate(random.randint(0,360))
#end

def contrast(img):
    enhancer = ImageEnhance.Contrast(img)

    factor = random.randint(80,120)/100
    im_output = enhancer.enhance(factor)

    return im_output
#end

def mirror(img):
    return ImageOps.mirror(img)
#end

def get_image(label,imgs):
    if label == "0-1":
        #Random aruco
        random_aruco_index = random.randint(0,99)
        #Non changing aruco
        # random_aruco_index = 0

        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_100)
        # second parameter is id number
        # last parameter is total image size
        img = aruco.drawMarker(aruco_dict, random_aruco_index, 100)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
        img = Image.fromarray(img)
    else:
        img = imgs.get(str(label))
    #end
    return img
#end

def superImpose(background,real,fake, labels,sq_size,i):
    #Superimposes all images in imgs ontop of background with no overlap, then saves the image into images folder.
    save_name = str(i)
    #Label file
    f = open("labels//"+save_name+".xml", "w")
    text_for_creation(f,save_name,sq_size,sq_size)

    #Randomly crop background and rescale back up to original size
    #Uses copy function so original background image can be used again
    new_img = section(background.copy(),sq_size)

    #Record coordinates of previously pasted images
    previous_pasted_coords = []

    for label in labels:
        #Check if a bounding box needs to be drawn for the sign
        bounding_box = False
        if label[0] in ["0","1","2"]:
            #Needs a bounding box as it is a sign
            img = get_image(label,real)
            bounding_box = True
        else:
            #Just a random image doesnt need a bounding box
            img = fake.get(label)
        #end

        #Rotate image
        theta = random.randint(0,360)
        rot = img.rotate(theta,expand=1)

        #Scale Image
        multipier = random.randint(10,300)/100
        rot.thumbnail((rot.size[0]*multipier,rot.size[1]*multipier), PIL.Image.ANTIALIAS)

        #Flip Image
        if random.choice([True, False]):
            rot = mirror(rot)
        #end

        #Contrast for overlayed image
        rot = contrast(rot)

        #Get widths and heights
        img_width,img_height = rot.size
        back_width, back_height = new_img.size


        #Get a random postion and ensure no overlapping images
        x_pos, y_pos = check_overlap(img_width,img_height,back_width,back_height,previous_pasted_coords)

        #If max recursion depth is reached dont include the picture
        if(x_pos < 0):
            break
        else:
            #Coords for bounding box, dont know why its like this but just take it.
            x1 = (x_pos+(img_width/2))/back_width
            y1 = (y_pos+(img_height/2))/back_width

            #Write label text file. Only save labels which need to be detected
            if bounding_box:
                save_to_text_file(f,label[0],x_pos,y_pos,img_width,img_height)  
            #end      

            #Superimpose image
            new_img.paste(rot,(x_pos,y_pos),rot)

            #Record pasted images coordinates
            previous_pasted_coords.append(Coords(x1,y1,img_width/back_width,img_height/back_height))
        #end
    #end

    #Close file
    f.write("\n</annotation>")
    f.close()

    shutil.copy("labels//"+save_name+".xml", "generated_images//labels//"+save_name+".xml")

    #Save image
    if i%100 == 0:
        print("Saving superimposed image and text file " + str(i))

    resize_image = (640,640)
    new_img.thumbnail(resize_image, Image.ANTIALIAS)
    
    #Add Gausian Noise
    new_img = noisy(new_img)

    #Change contrast
    new_img = contrast(new_img)

    new_img.convert('RGB').save('generated_images//images//'+save_name+'.jpg')
#end

#Stuff to include
#Generate image similar to the black and white block one

#Finished stuff
#Different backgrounds
#Random number of signs in image
#Multiple sign can appear multiple times
#Similar signs
#Stop te signs from overlapping

#Remove all images previously in the folder
files = glob.glob('generated_images/images/*')
for f in files:
    os.remove(f)
#Remove all images previously in the folder
files = glob.glob('generated_images/labels/*')
for f in files:
    os.remove(f)

#############################################################################################################################
#Change these lines to adjust the resolution of the model and the generated size
background_size = (640,640)   #Inital background load in size, want this to be a high res image so when sectioned it will still look good.
overlay_size = (100,int(100*sqrt(2)))    #Image which is overlayed will be a random size between 100 and this size. This needs to be smaller than generated_size
generated_size = 640   #Output size of the image, specificies how big a section is from the background.

num_images_to_genereate = 1000  #Number of images to generate

#############################################################################################################################


#Load in 
floors, real_signs, fake_signs = load_images(background_size,overlay_size)

for i in range(num_images_to_genereate):
    num_to_impose = 3
    signs_to_impose = []

    real_impose = range(0,random.randint(1,num_to_impose))
    for n in real_impose:
        signs_to_impose.append(random.choice(list(real_signs.keys())))
    #end

    fake_impose = range(0,random.randint(1,num_to_impose))
    for n in fake_impose:
        signs_to_impose.append(random.choice(list(fake_signs.keys())))
    #end

    #Superimposes image, saves output and bounding box in YOLO format.
    superImpose(random.choice(floors),real_signs,fake_signs,signs_to_impose,generated_size,i)
#end