import math
import sys

import data
from model import FCNet
from model import PilotNet
from model import Modified_Nvidia_Netwrok

BATCH_SIZE = 256
AUGMENT_DATA = True
FIX_DISTRIBUTION = True

if __name__ == "__main__":
    
    NUMBER_EOPCHS = 20
    
    # sys.argv[0] is the dir of the runned file
    if len(sys.argv) > 1:
        NUMBER_EOPCHS = int(sys.argv[1])
    
    ## Load the image paths and steering data into memory from 
    # the csv file. Images are not actually loaded until needed
    # by the generator when making batches
    data_path = '/media/ubuntu16/Documents/datasets/CarND/' 
    csv_path = data_path + 'driving_log.csv'
    
    training_set, validation_set = data.load_data_sets(csv_path)

    print("EPOCHS: {}".format(str(NUMBER_EOPCHS)))
    print("Training Set Size: {}".format(str(len(training_set))))
    print("Valization Set Size: {}".format(str(len(validation_set))))
    print("Batch Size: {}".format(str(BATCH_SIZE)))

    
    ## perform data augmentation if necessary
    if AUGMENT_DATA:
        training_set = data.augment_dataset(training_set, FIX_DISTRIBUTION)
        print("Training set size now: {}".format(str(len(training_set))))

    ## data generator
    train_generator = data.get_generator(training_set, BATCH_SIZE)
    valid_generator = data.get_generator(validation_set, BATCH_SIZE)


    ##-- Modified Nvidia Network --##
    #model = Modified_Nvidia_Netwrok()

    ##-- FCNet --##
    #model = FCNet()

    ##-- PilotNet --##
    model = PilotNet()

    ##
    training_steps = math.ceil(len(training_set)/BATCH_SIZE)
    validation_steps = math.ceil(len(validation_set)/BATCH_SIZE)

    ## train
    model.fit(train_generator, valid_generator, training_steps, validation_steps, 
              NUMBER_EOPCHS)
    
    ## save model
    #model.save_model('modified_nvidia_model.h5')

    ## evaluate model


