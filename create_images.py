from PIL import Image
import PIL
import random
from math import cos,sin
import shutil

from os import listdir
from os.path import isfile, join
import glob, os


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
    #Load base images
    # signs = ["chems.png","aruco.png","goods.png",'sign1.png','sign2.png','sign3.png','sign4.png','sign5.png','sign6.png','sign7.png','sign8.png','sign9.png']
    signs = [f for f in listdir('images/') if isfile(join('images/', f))]
    # background_size = (3000,3000)
    # signs_size = (300,300)

    
    signs_img = {}
    for i,sign in enumerate(signs):
        signs_img[str(i)] = Image.open('images/'+sign).resize(signs_size).convert('RGBA')
    #end

    # backgrounds = ['floor1.png','floor2.png','floor3.png','floor4.png','floor5.png','floor6.png','floor7.png']
    backgrounds = [f for f in listdir('backgrounds/') if isfile(join('backgrounds/', f))]
    output_backgrounds = []
    for background in backgrounds:
        output_backgrounds.append(Image.open('backgrounds/'+background).resize(background_size).convert('RGBA'))
    #end

    return output_backgrounds, signs_img
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
    
    labels = {"0":"Chemical_Sign",
                "1":"Aruco",
                "2":"Dangerous_Goods"}

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

def superImpose(background, imgs, labels,sq_size,labels_list,i):
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

        #Rotate image
        theta = random.randint(0,360)
        rot = imgs.get(str(label)).rotate(theta,expand=1)

        #Scale Image
        new_size = random.randint(100,rot.size[0])
        rot.thumbnail((new_size,new_size), PIL.Image.ANTIALIAS)

        #Get widths and heights
        img_width,img_height = rot.size
        back_width, back_height = new_img.size

        #Get a random postion and ensure no overlapping images
        x_pos, y_pos = check_overlap(img_width,img_height,back_width,back_height,previous_pasted_coords)

        #If max recursion depth is reached dont include the picture
        if(x_pos < 0):
            # print("Reach max recusion")
            break
        else:
            #Coords for bounding box, dont know why its like this but just take it.
            x1 = (x_pos+(img_width/2))/back_width
            y1 = (y_pos+(img_height/2))/back_width

            #Write label text file. Only save labels which need to be detected
            if label in labels_list:
                save_to_text_file(f,label,x_pos,y_pos,img_width,img_height)  
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

    resize_image = (300,300)
    new_img.thumbnail(resize_image, Image.ANTIALIAS)
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

#############################################################################################################################
#Change these lines to adjust the resolution of the model and the generated size
background_size = (3000,3000)   #Inital background load in size, want this to be a high res image so when sectioned it will still look good.
overlay_size = (300,300)    #Image which is overlayed will be a random size between 100 and this size. This needs to be smaller than generated_size
generated_size = 1000   #Output size of the image, specificies how big a section is from the background.

num_images_to_genereate = 3000  #Number of images to generate

labels_list = [0,1,2]   #Add more labels as needed. Need to make sure that the overlayed images are in order at the top of the images folder. so the first 3 by alphabetical will be the labelled ones.
#############################################################################################################################


#Load in 
floors, signs = load_images(background_size,overlay_size)

for i in range(num_images_to_genereate):
    #A random number of signs to impose between 1 and 5.
    #Then randomly selects between the 3 different signs
    num_to_impose = 5
    signs_to_impose = []
    for n in range(0,random.randint(1,num_to_impose)):
        signs_to_impose.append(random.randint(0,len(signs)-1))
    #end

    #Superimposes image, saves output and bounding box in YOLO format.
    superImpose(random.choice(floors),signs,signs_to_impose,generated_size,labels_list,i)
#end

