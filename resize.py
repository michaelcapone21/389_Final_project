from unittest import result
from PIL import Image
import os
import matplotlib.pyplot as plt

#This method comes from https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
def resize():
     for dir in (os.listdir('./datasets')):
        for file in (os.listdir('./datasets/'+dir)):
            img = Image.open( './datasets/'+dir+'/'+ file)
            basewidth = 28 #this is the size of width in pixels
            wpercent = (basewidth/float(img.size[0])) #perserves the aspect ratio
            hsize = int((float(img.size[1])*float(wpercent)))
            img = img.resize((basewidth,hsize), Image.ANTIALIAS)
            img.convert("RGBA").save('./datasets/'+dir+'/'+ file) #this changes the shape of the data. Some emojis are (1,28,28) change to (4,28,28)




def main():
    resize()

if __name__ == "__main__":
    main()