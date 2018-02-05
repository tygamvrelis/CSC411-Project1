Instructions:

See the report for system specifications (i.e. Python version and libraries required). Note that Images.zip is for the images in the LaTeX report.


Instructions for running code (please read to the end of this file prior to performing any of the steps!):

1. Place faces.py, get_data.py, Part3.py, Part5.py, and Part6.py in the same directory

2. Create a folder called "datasets" relative to this directory
	2.1. In this directory, place the female subset of FaceScrub in a text file called "female_faces.txt"
	2.2 In this directory, place the male subset of FaceScrub in a text file called "male_faces.txt"

3. In faces.py, change the line in main involving os.chdir so that your desired working directory is specified

4. Run faces.py



IF you do not desire to perform the image downloading, then do the following before step 4:

1. Unzip the images in cropped.zip into a folder called "cropped" in the current working directory

2. In faces.py, comment out all the lines in part 1 and part 2 of the project1 function. Part 1 and part 2 make use of the uncropped pictures which were not submitted to Markus due to their large size.

After performing these 2 steps, parts 3 through 8 will run properly.