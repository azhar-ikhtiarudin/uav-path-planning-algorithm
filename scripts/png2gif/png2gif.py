import argparse
import imageio
import imageio.v3 as iio
from pathlib import Path
from PIL import Image
from natsort import os_sorted
import os

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-p", "--path", help="path to folder containing images")
    argParser.add_argument("-o", "--output", help="output file name")
    argParser.add_argument("-fps", "--fps", help="frames per second", default=10)

    args = argParser.parse_args()
    images = list()
    
    print("reading images from "+args.path)
    for subdir, dirs, files in os.walk(args.path):
        for file in os_sorted(files):
            filepath = subdir + os.sep + file
            images.append(Image.fromarray(iio.imread(filepath)).resize((1080, 720), resample=Image.NEAREST))
        
    

    if args.output is None:
        args.output = args.path+".gif"
    kwargs_write = {}
    print("writing gif to "+args.output)
    imageio.mimsave(args.output, images, 'GIF-PIL', **kwargs_write)
   
   

if __name__ == "__main__":
   main()