#Author: Lang LIN
#Usage: 1.python Convert_PDF_To_Image.py
#       2.python Convert_PDF_To_Image.py --pdf_directory="directory for input pdf files" --img_directory="wishing directory for output image files"
#Effect: convert all .pdf(no matter .PDF or .Pdf of .pdf) files in input pdf directory into image files in output image directory.
#        default pdf directory is "./pdf/", default image directory is "./images/"

from pdf2image import convert_from_path
import glob
import argparse
import os


parser = argparse.ArgumentParser(description='Enter directory for input pdf files & directory for output image files')
parser.add_argument('-p','--pdf_directory',default='./pdf/',help='directory for input pdf files,default is ./pdf/')
parser.add_argument('-i','--img_directory',default='./images/',help='directory for output img files,default is ./images/')

args = vars(parser.parse_args())
output_path = os.path.dirname(args['img_directory'])

try:
    os.stat(output_path)
except:
    os.mkdir(output_path)


file_pattern = args['pdf_directory'] + '/*'
files = glob.glob(file_pattern)
sorted_files = sorted(files)
print(str(len(sorted_files)))
for file in sorted_files:
    if ".pdf" in file.lower():
        pages = convert_from_path(file)
        count = 0
        file_name = file[:-4].replace(args['pdf_directory'],args['img_directory'])
        if len(pages) == 1:
            output_file = file_name + ".jpg"
            pages[0].save(output_file)
            print("finished output image file of pdf file " + file_name + ".pdf.")
        else:
            for page in pages:
                count = count + 1
                output_file = file_name + "-" + str(count) + ".jpg"
                page.save(output_file)
            print("finished output image file of pdf file " + file_name + ".pdf, it has " + str(len(pages)) + " pages.")
