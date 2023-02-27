import logging
import re
import os
import time
import xlwt
import datetime

'''
you can get the output in the filename output 
'''


def get_logger(filename, verbosity=1, name=None):
    ## level 的主要目的是用来调整logging的级别
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def log_to_excel(log_path, output_name):
    path = log_path

    downloadtime = time.strftime('%Y_%m_%d')
    filename = output_name + downloadtime + '.xls'
    workbook = xlwt.Workbook(encoding='ascii')
    worksheet = workbook.add_sheet('Comment')
    workbook.save(filename)

    font = xlwt.Font()  # Create the Font
    font.bold = True
    font.colour_index = 1
    font.height = 0x00C8  # C8 in Hex (in decimal) = 10 points in height.
    style = xlwt.XFStyle()  # Create the Style
    style.font = font  # Apply the Font to the Style
    pattern = xlwt.Pattern()  # Create the Pattern
    pattern.pattern = xlwt.Pattern.SOLID_PATTERN  # May be: NO_PATTERN, SOLID_PATTERN, or 0x00 through 0x12
    pattern.pattern_fore_colour = 23  # May be: 8 through 63. 0 = Black, 1 = White, 2 = Red, 3 = Green, 4 = Blue, 5 = Yellow, 6 = Magenta, 7 = Cyan, 16 = Maroon, 17 = Dark Green, 18 = Dark Blue, 19 = Dark Yellow , almost brown), 20 = Dark Magenta, 21 = Teal, 22 = Light Gray, 23 = Dark Gray, the list goes on...
    style.pattern = pattern  # Add Pattern to Style
    row = 1
    for root, dirs, files in os.walk(path):
        for file in files:
            log_file = os.path.join(path, file)

            fp = open(log_file, encoding='utf-8')

            for line in fp.readlines():
                if "push messages count" in line:
                    n = re.search(r'count:', line).span()[1]
                    count = line[n:n + 5].split(',')[0]
                    print('Count: ' + line[n:n + 5].split(',')[0])
                    k = re.search(r'seId', line).span()[1]
                    sendTime = line[k:].strip('\\').split('\"')[2].strip('\\')
                    sendTime = datetime.datetime.strptime(sendTime, "%Y-%m-%d %H:%M:%S")
                    # t = sendTime.timetuple()
                    # timeStamp = int(time.mktime(t))
                    # # h = int(sendTime[1].split(':')[0]) + 8
                    # dateArray = datetime.datetime.utcfromtimestamp(timeStamp / 100 + 2880)
                    # sendTime = dateArray.strftime("%Y-%m-%d %H:%M:%S")
                    sendTime = (sendTime + datetime.timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S")

                    print('SendTime: ' + sendTime)
                    sendTime1 = datetime.datetime.strptime(sendTime, "%Y-%m-%d %H:%M:%S")
                    receTime = line[1:].split(r']')[0].split('.')[0]
                    receTime2 = datetime.datetime.strptime(receTime, "%Y-%m-%d %H:%M:%S")
                    # print ('ReceTime: '+ receTime)
                    delateTime = (receTime2 - sendTime1).seconds
                    worksheet.write(row, 0, count)
                    worksheet.write(row, 1, sendTime)
                    worksheet.write(row, 2, receTime)
                    worksheet.write(row, 3, delateTime)
                    row = row + 1

    worksheet.write(0, 0, "count", style)
    worksheet.write(0, 1, "sendTime", style)
    worksheet.write(0, 2, "receTime", style)
    worksheet.write(0, 3, "delateTime", style)
    worksheet.col(0).width = 2222
    worksheet.col(1).width = 5555
    worksheet.col(2).width = 5555
    worksheet.col(3).width = 5555
    workbook.save(filename)
