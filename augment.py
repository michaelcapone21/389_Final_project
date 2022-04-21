from unittest import result
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def rotate():
     for dir in (os.listdir('./datasets')):
        if dir == 'Augmented':
            continue
        for file in tqdm((os.listdir('./datasets/'+dir))):
            img = Image.open( './datasets/'+dir+'/'+ file)
            img =img.rotate(90)
            img.save('./datasets/Augmented/'+dir+'_rotated_90_'+file) #this changes the shape of the data. Some emojis are (1,28,28) change to (4,28,28)

def merge():
     for dir in (os.listdir('./datasets')):
        img = None
        if dir == 'Augmented':
            continue
        for file in tqdm((os.listdir('./datasets/'+dir))):
            out = img
            if(img != None):
                out = Image.blend(img,Image.open( './datasets/'+dir+'/'+ file), .5)
            img = Image.open( './datasets/'+dir+'/'+ file)
            if(out != None):
                out.save('./datasets/Augmented/'+dir+'_blur_'+file)
                
def mergeAll():
     for dir in (os.listdir('./datasets')):
        if dir == 'Augmented':
            continue
        for file in((os.listdir('./datasets/'+dir))):
            out = Image.open( './datasets/'+dir+'/'+ file)
            for file in tqdm((os.listdir('./datasets/'+dir))):
                img = Image.open( './datasets/'+dir+'/'+ file)
                if(img != out):
                    out = Image.blend(img,out, .5)
                out.save('./datasets/Augmented/'+dir+'_blur_all_'+file)
                # out.show()


                
            

def main():
    # rotate()
    # merge()
    mergeAll()
    print('l')
if __name__ == "__main__":
    main()

