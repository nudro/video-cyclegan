# READ ME


### Purpose: Translates a video to style of "summer2winter" using pretrained CycleGAN model.

### Usage:

    ```
    python main.py --name summer2winter_yosemite_pretrained --model test --preprocess none --no_dropout

    ```

- `videos` are where the "real" videos should be stored. These are the videos you want to translate from. 

### Outputs:

- "result_frames" dir. Some samples are provided.
- translated frames written to "result_frames"
- output_part1.mp4 saved to root
- for all of the above ^^ you will want to manually change the names for new videos.


### NOTES:

(1) I modified the arg of '--dataroot' in options/base_options.py = False, in order to use a video, not a datafolder.
(2) The current "result_frames" dir is empty. All previous images written to this dir were zipped up and  provided, alongside the script due to size constraints.

### OUTPUT VID RES

CycleGAN only translates on frames/images with dims 256 x 256. However, when resizing the frame and writing to video, you can use 512 x 512, 1024 x 1024, or the actual video size. output_part1, 2, and 3.mp4 vids are 512 x 512. output_part4.mp4 is 1024 x 1024 and output_part1_trusize.mp4 is the vid's regular dims.

### MANUALLY:

See to do below. Currently, you need to manually change the 'result_frames' dir, video filepath under cv2.VideoCapture, and path for cv2.VideoWriter to whatever video you like. If you chose to write frames to image using result_frames, don't forget to change the dir name, otherwise you'll be writing over all images.

### TO DO:

- Add to the existing opt so that you can pass:
    - [ ] frames=True if you want to write frames/not
    - [ ] string for filepath of video to be captured and concurrently, video to be written (output)
    - [ ] string for filepath of "result_frames" dir. Remember, if you run different vids, currently
    you have to rename result_frames brute force

### OTHER IDEAS:

- From a post-processing perspective, to obtain crispness, as these models are succumb to blurriness, you might want to consider using a super-resolution algorithm.
- May want to try other pretrained checkpoints such as `day2night`.
- `fps` and `size` may influence the quality of generated frames.
- May want to try other videos. Example real videos are clips from drone footage pulled from YouTube.

"""
