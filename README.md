# RoboPainter
This program uses a turns a reference image into a svg image file.

## Required Dependencies
- python 3.9+
- potrace -> This has to be manually installed seperately and added into PATH

**Installing Potrace**
1. Goto potrace's website download page [website](http://potrace.sourceforge.net/#downloading)
2. Download a precompiled distribution for your os.
3. Extract the contents of the archive to a place of your choice. eg. Documents folder
4. Add the path of the potrace executable in the extracted archive to PATH.
5. Check you have installed potrace properly by opening command prompt and entering potrace -h

    This should bring up the help menu for the potrace cli


**The dependencies below will be installed automatically or can be installed with pip**

- BeautifulSoup4
- opencv-python
- scipy
- tqdm
- numpy
- fire
- lxml