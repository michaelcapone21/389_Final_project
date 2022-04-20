from unittest import result
from PIL import Image
import os
import matplotlib.pyplot as plt


def rotate():
     for dir in (os.listdir('./datasets')):
        for file in (os.listdir('./datasets/'+dir)):
            img = Image.open( './datasets/'+dir+'/'+ file)
            img.transpose(Image.ROTATE_90)
            img.save('./datasets/Augmented/'+ file+ 'rotated_45') #this changes the shape of the data. Some emojis are (1,28,28) change to (4,28,28)



def main():
    rotate()
if __name__ == "__main__":
    main()

