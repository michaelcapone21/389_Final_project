from unittest import result
from PIL import Image
import os
import matplotlib.pyplot as plt



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
                
# def mergeAll():
#      for dir in (os.listdir('./datasets')):
#         if dir[len(dir)-3:] == 'ged': # ged not merged bc of folders like Apple that are less than length 6
#             continue
#         for file1 in((os.listdir('./datasets/'+dir))):
#             out = Image.open( './datasets/'+dir+'/'+ file1)
#             #i = -1
#             for file2 in ((os.listdir('./datasets/'+dir))):
#                 #i += 1
#                 # if i == 0:
#                 #     continue
#                 if file1[]
#                 img = Image.open( './datasets/'+dir+'/'+ file2)
#                 if(img != out):
#                     out = Image.blend(img,out, .5)
#                 out.save('./datasets/'+ dir + '_merged/'+dir+'_merge_'+file1[:len(file1)-4] +'_'+ file2 )


def mergeAll():
    for dir in (os.listdir('./datasets')):
        if dir[len(dir)-3:] == 'ged':
            continue
        for i in range(1,len(os.listdir('./datasets/'+dir))+1):
            image_a = Image.open('./datasets/'+dir+'/'+str(i)+'.png')
            for j in range(i+1, len(os.listdir('./datasets/'+dir))+1):
                image_b = Image.open('./datasets/'+dir+'/'+str(j)+'.png')
                result = Image.blend(image_a, image_b, .5)
                result.save('./datasets/'+dir+'_merged/'+dir+'_merge_'+str(i)+'_'+str(j)+'.png')









                
            

def main():
    # rotate()
    # merge()
    mergeAll()
    print('l')
if __name__ == "__main__":
    main()

