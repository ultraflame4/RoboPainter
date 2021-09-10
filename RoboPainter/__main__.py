

import math
import os
import shutil
import bs4
import cv2
import numpy as np
from scipy import stats
import re
from tqdm import tqdm
import fire
from .imagetiles import *



class PixelFlattener:

    MaxColorDiff :int = 50
    MaxAvgDiff :int = 50
    SurroundRadius = 2

    @staticmethod
    def cleanUp(im:np.ndarray):
        # Removes random lonely pixels etc.
        total_pixelCount = (PixelFlattener.SurroundRadius*2+1)**2
        height, width, depth = im.shape

        bufferIm = im.copy()
        print("\nCleaning up...")
        for y in tqdm(range(height)):
            for x in range(width):
                center = im[y, x]

                sp = im[y-PixelFlattener.SurroundRadius:y+PixelFlattener.SurroundRadius+1,x-PixelFlattener.SurroundRadius:x+PixelFlattener.SurroundRadius+1]
                surroundingPixVals = sp.reshape(-1,sp.shape[-1])


                if len(surroundingPixVals) < 1:
                    bufferIm[y, x] = center
                    continue

                percentage = np.mean(surroundingPixVals == np.full(surroundingPixVals.shape, center))
                if percentage < .5:
                    bufferIm[y, x] = stats.mode(surroundingPixVals)[0]

        return bufferIm


    @staticmethod
    def flatten(img:np.ndarray,tl):
        # rounds pixels values in an area to an average value. like image compression
        height, width, depth = img.shape

        bufferIm = img.copy()


        lastAverages = [[0,0,0]]


        def worker(im:np.ndarray, offsetX, offsetY,tileLength,pbar:tqdm):
            for y in range(tileLength):
                oy = y + offsetY
                for x in range(tileLength):
                    ox=x+offsetX

                    if ox >= im.shape[1] or oy >= im.shape[0]:
                        pbar.update(1)
                        continue

                    center = im[oy,ox]

                    if ox >= width-1:
                        right=center
                    else:
                        right = im[oy,ox+1]


                    color_diff = math.dist(right,center)
                    if color_diff < PixelFlattener.MaxColorDiff:
                        avg = (right/2+center/2)

                        # check for other similar averages before using new average
                        avgDiffs = [math.dist(avg,a) for a in lastAverages]
                        bestDist = min(avgDiffs)
                        if bestDist < PixelFlattener.MaxAvgDiff:
                            bestMatch = lastAverages[avgDiffs.index(bestDist)]
                            bufferIm[oy, ox] = bestMatch

                        else: # if bestMatch still over threshold, create new average
                            lastAverages.append(avg)
                            bufferIm[oy, ox] = avg

                    pbar.update(1)

        splitimager(img,worker,tl)

        return bufferIm


class BuildPaths:
    @staticmethod
    def getPixelSeperatorBuildDir(name):
        build_dir = f"./build/{name}-variants"
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        return build_dir


    @staticmethod
    def getPotraceBuildDir(name):
        build_dir = f"./build/{name}-potrace"
        if not os.path.exists(build_dir):
            os.mkdir(build_dir)
        return build_dir





class PixelSeperator:
    # Seprates each pixel value into its own image and saves them
    @staticmethod
    def seperate(name:str,im:np.ndarray):
        print(f"Creating svg files for each pixel variant for band: {name}...")
        build_dir = BuildPaths.getPixelSeperatorBuildDir(name)

        print("Clearing existing files...")
        for file in os.listdir(build_dir):
            os.remove(f"{build_dir}/{file}")


        all_pix_vals = np.unique(im)
        print("Number of pixels variants: ",len(all_pix_vals))
        print("Generating variant files for band:",name)
        for variant in tqdm(all_pix_vals):
            locations = np.where(im == variant)
            if len(locations[0]) < 2:
                continue
            variantIm = np.full(im.shape,255,im.dtype)
            for yIndex,yPosition in enumerate(locations[0]):
                variantIm[yPosition,locations[1][yIndex]]=0
            cv2.imwrite(f"{build_dir}/pixel_varient_{variant}.bmp",variantIm)



class PotraceConverter:
    Path = "potrace" # potrace is in path
    flags = "-b svg"
    strokeWidth=2
    @staticmethod
    def convert(filepath,outpath):
        return os.system(f"{PotraceConverter.Path} {filepath} -o {outpath} {PotraceConverter.flags}")

    @staticmethod
    def convertAll(name):
        src_path = BuildPaths.getPixelSeperatorBuildDir(name)
        build_path = BuildPaths.getPotraceBuildDir(name)

        print("Clearing existing files...")
        for file in os.listdir(build_path):
            os.remove(f"{build_path}/{file}")

        print("Potrace converting for band",name)
        for file in tqdm(os.listdir(src_path)):
            PotraceConverter.convert(f"{src_path}/{file}",f"{build_path}/{file}.svg")


    @staticmethod
    def stackFiles(name,band):
        print(f"Stacking pixel variants svg files for band: {name}...")
        build_path = BuildPaths.getPotraceBuildDir(name)
        c=[0,0,0]
        c[band]=1
        c = tuple(c)

        with open(f"{build_path}/{os.listdir(build_path)[0]}","r") as f:
            baseSoup = bs4.BeautifulSoup(f.read(),"xml")
            first=True

            for file in os.listdir(build_path):
                if first:
                    first=False
                    continue
                with open(f"{build_path}/{file}", "r") as fi:
                    fileSoup = bs4.BeautifulSoup(fi.read(),"xml")
                    t=  fileSoup.g
                    color = re.findall(r"(?<=_)\d*",file)# use regex to get color from filename
                    if len(color) > 1:
                        color_ = int(color[-1])
                        color_string = f"rgb({c[2]*color_},{c[1]*color_},{c[0]*color_})"
                        t["fill"]=color_string
                        t["stroke"] = color_string
                        t["stroke-width"] = str(PotraceConverter.strokeWidth)

                        baseSoup.svg.append(t)

            with open(f"{build_path}/final.svg","w") as ff:
                ff.write(str(baseSoup.prettify()))



def convSepStack(name,img,band):
    print("-Working on",name,"...-")
    im=img[:,:,band]
    PixelSeperator.seperate(name,im)
    PotraceConverter.convertAll(name)

    PotraceConverter.stackFiles(name,band)
    print("-Finised",name,"...-")

def resize(scale_percent:int,img:np.ndarray):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)





def combineFiles(name1,name2,name3,out):
    with open(f"{BuildPaths.getPotraceBuildDir(name1)}/final.svg","r") as f:
        name1_soup = bs4.BeautifulSoup(f.read(),"xml")
        with open(f"{BuildPaths.getPotraceBuildDir(name2)}/final.svg", "r") as f2:
            name2_soup = bs4.BeautifulSoup(f2.read(), "xml")
            allGroups = name2_soup.svg.findAll("g",recursive=False)
            for g in allGroups:
                name1_soup.svg.append(g)

            with open(f"{BuildPaths.getPotraceBuildDir(name3)}/final.svg", "r") as f3:
                name3_soup = bs4.BeautifulSoup(f3.read(), "xml")
                allGroups = name3_soup.svg.findAll("g", recursive=False)
                for g in allGroups:
                    name1_soup.svg.append(g)

        styleTag = name1_soup.new_tag("style")
        styleTag.string="path,g { mix-blend-mode: screen; }"
        name1_soup.svg.insert(0,styleTag)

        with open(out, "w") as ff:
            ff.write(str(name1_soup.prettify()))
            print("Finished writing to file at",out)



def paint(input_filepath,out_path="./out.svg",dump_bands=False,delete_build=True,fMaxColorDiff=50,fMaxAvgDiff=50,fAvgRadius=2,borderSize=5,tileLength=300):
    """
    :param input_filepath: Input path for the file to convert

    :param out_path: Output path for the output file

    :param dump_bands: Debug option True/False dumps the rgb bands of image after image preparations

    :param delete_build: Deletes the temporary automatically created build directory that contains files created and used by the program.

    :param fMaxColorDiff: Image preparation option. Maximum difference between the orignal colors before it cannot be assumed and averaged as the same color.

    :param fMaxAvgDiff: Image preparation option. Maximum difference between the new average color and prexisting averages before using new average

    :param fAvgRadius: Image preparation option. Search radius for caculating average

    :param borderSize: Border size of each group path
    :return:
    """
    print("Robopainter 10")
    FINAL_OUTPUT_PATH = out_path
    PixelFlattener.MaxColorDiff=fMaxColorDiff
    PixelFlattener.MaxAvgDiff=fMaxAvgDiff
    PixelFlattener.SurroundRadius=fAvgRadius
    PotraceConverter.strokeWidth=borderSize

    if not os.path.exists("./build"):
        os.mkdir("./build")


    print("Reading image...")
    image = cv2.imread(input_filepath)
    print("Beginning image preparations...")
    flattened_im = PixelFlattener.flatten(image,tileLength)
    cleaned_im=PixelFlattener.cleanUp(flattened_im)
    print("Finished image preparations.")


    if (dump_bands):
        print("Dumping image bands...")
        cv2.imwrite("./build/blue.bmp",cleaned_im[:,:,0])
        cv2.imwrite("./build/green.bmp",cleaned_im[:,:,1])
        cv2.imwrite("./build/red.bmp",cleaned_im[:,:,2])

    print("Sending image channels to potrace...")
    convSepStack("blue",cleaned_im,0)
    convSepStack("green",cleaned_im,1)
    convSepStack("red",cleaned_im,2)

    print("Merging resulting svg files...")
    combineFiles("red","green","blue",FINAL_OUTPUT_PATH)
    print(f"Finished merging, output file at {FINAL_OUTPUT_PATH}")
    if delete_build:
        print('Clean up: removing temporary working build directory..')
        shutil.rmtree("./build")
        print("Finished cleanup. Done!")

if __name__ == "__main__":
    fire.Fire(paint)
