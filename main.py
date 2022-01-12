import os
from options.test_options import TestOptions #calls options.test_options.py
from data import create_dataset
from models import create_model
import cv2
import torch
import numpy as np
from util import*


# Adapted from https://github.com/bensantos/webcam-CycleGAN/blob/master/webcam.py

if __name__ == '__main__':

    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    # Makes dir to store each frame from the video
    # Manually change this if you are writing to a new video, otherwise it'll write over contents
    os.makedirs("./result_frames", exist_ok=True)

    print("Opening the video...")
    vc = cv2.VideoCapture("./videos/england_3.mp4") # TODO: Pass arg to process any video

    if not vc.isOpened():
        raise IOError("Cannot open video.")

    ##########################
    # video and codec params
    ##########################

    length = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    currentFrame = 0

    # Select the current video dims, or make your own (commented out below - 512x512 or 1024x1024 for example)

    size = (
        int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )


    #size = (512,512) #Use resized frame size, otherwise it will not write to file!
    #size = (1024,1024)

    print("size:", size)

    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    output = cv2.VideoWriter('output_part1.mp4', fourcc, 25.0, size) #TODO: Pass arg to name it, need to bypass CycleGAN TestOpions()


    # Ref: https://towardsdatascience.com/using-cyclegan-to-perform-style-transfer-on-a-webcam-244142effe7f
    # start an infinite loop and keep reading frames from the webcam until we encounter a keyboard interrupt
    data = {"A": None, "A_paths": None}

    while vc.isOpened():
        ret, frame = vc.read()

        if ret==True: #(frame read succesfuly)
            currentFrame += 1
            print("currentFrame:", currentFrame)

            #resize frame - CycleGAN takes 256x256 only
            frame = cv2.resize(frame, (256,256), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            #model wants BxCxHxW
            #gives it a dimension for batch size
            frame = np.array([frame])
            #permute it to get: BxCxHxW
            frame = frame.transpose([0,3,1,2])

            #convert numpy array to tensor - this is in line with the /testA dataset
            data['A'] = torch.FloatTensor(frame)

            model.set_input(data)  # unpack data from data loader
            model.test()

            #get only generated image - indexing dictionary for "fake" key
            result_image = model.get_current_visuals()['fake']

            #use tensor2im provided by util file
            result_image = util.tensor2im(result_image)
            result_image = cv2.cvtColor(np.array(result_image), cv2.COLOR_BGR2RGB)

            """
            ! Very important: The size you selected for the writer, must exactly
            match the size of image resizing. Otherwise it will not write to output.
            """
            #result_image = cv2.resize(result_image, (512, 512))
            # larger res option:
            result_image = cv2.resize(result_image, size)

            # write to file
            #cv2.imshow("summer2winter", result_image) #<- cant do on server
            #output.write(result_image)

            # now write to frame
            cv2.imwrite("./result_frames/result_image_{}.png".format(currentFrame), result_image)

            output.write(result_image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        else:
            break


# ========//
# When everything is done, release the capture
vc.release()
output.release()
cv2.destroyAllWindows()
