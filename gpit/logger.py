#Logger module is part of the gPit software  (https://github.com/GuilhermeMene/gPit)
#Copyright (c) 2023 gPit Developers 
#Distributed under the terms of MIT License (https://github.com/GuilhermeMene/gPit/blob/main/LICENSE)

import datetime
import gpit
import os


if gpit.LOGPATH == '':
    print("The gPit will be save all logs in defautl directory")

    directory = os.getcwd()

    filepath = os.path.join(directory, 'Log.txt')
else:
    filepath = os.path.join(gpit.LOGPATH, 'Log.txt')

def datalogger(datatolog:str):

    time = str(datetime.datetime.now())

    print(datatolog)

    with open(filepath, 'a') as file: 
        file.write(f"{time} - {datatolog} \n")
        file.close

