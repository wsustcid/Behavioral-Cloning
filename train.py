import math
import sys

import data
from model import Modified_Nvidia_Netwrok

BATCH_SIZE = 256
AUGMENT_DATA = False
FIX_DISTRIBUTION = True

if __name__ == "__main__":
    
    NUMBER_EOPCHS = 10
    
    # sys.argv[0] is the dir of the runned file
    if len(sys.argv) > 1:
        NUMBER_EOPCHS = int(sys.argv[1])
    
    ## Load the image paths and steering data into memory from 
    # the csv file. Images are not actually loaded until needed
    # by the generator when making batches
    data_path = '/media/ubuntu16/新加卷/Self-Driving/datasets/carnd/' 
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


    ## Modified Nvidia Network
    #model_path = 
    model = Modified_Nvidia_Netwrok('model-{epoch:02d}.h5')

    ##
    training_steps = math.ceil(len(training_set)/BATCH_SIZE)
    validation_steps = math.ceil(len(validation_set)/BATCH_SIZE)

    ## train
    model. fit(train_generator, valid_generator, training_steps, validation_steps, NUMBER_EOPCHS)


