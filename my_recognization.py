import numpy as np
import cv2 as cv
from functools import partial
from model_class import PPHumanSeg
import os

class filter_creator(object):
    """
    This class is the used to create all the filters that are going to be used in the project.
    Args:
        object (object): base class from which we are inheriting.    
    """

    def __init__(self):
        self.pp_human_model = PPHumanSeg(modelPath=r'Video_filter_JN/models/human_segmentation_pphumanseg_2023mar.onnx',
                                    backendId=cv.dnn.DNN_BACKEND_OPENCV, targetId=cv.dnn.DNN_TARGET_CPU)
        # self.yunet_model = YuNet('model/face_detection_yunet_2023mar.onnx',backendId=cv.dnn.DNN_BACKEND_OPENCV, targetId=cv.dnn.DNN_TARGET_CPU)
        
        # Configure the model
        self._model = self.pp_human_model
        self.weight = 0.6
        self.window = ''
        self.filter_selected = 1
        self.enable_fps = True
        self.background_image_index = 0
        
        self.background_image_path_list = []
        self.kernel_size = 11
        self.func_name = self.create_filter
        self.background_enable = False
        
        # key dict
        self.key_mapper = {
            ord('d'): self.d_press,
            ord('f'): self.f_press,
            ord('n'): self.n_press,
            ord('p'): self.p_press,
            ord('1'): self.one_press,
            ord('2'): self.two_press,
            ord('3'): self.three_press,
            ord('4'): self.four_press,
            ord('5'): self.five_press,
            ord('r'): self.rotate_background_image,
            ord('k'): self.change_kernel_size
        }

        #Load all the backgorund images
        self.directory = r'Video_filter_JN\background_images\result'
        for filename in os.listdir(self.directory ):
            f = os.path.join(self.directory , filename)
            # checking if it is a file
            if os.path.isfile(f):
                self.background_image_path_list.append(f)
        
        self.background_image = cv.imread(self.background_image_path_list[self.background_image_index])
            
    def d_press(self):
        quit()
        # return "quit"
    def f_press(self):
        self.enable_fps = not self.enable_fps
    def n_press(self):
        self.filter_selected = self.filter_selected + 1
        if self.filter_selected > 5:
            self.filter_selected = 1
    def p_press(self):    
        self.filter_selected = self.filter_selected - 1
        if self.filter_selected < 1:
            self.filter_selected = 5
    def one_press(self):
        self.filter_selected = 1
    def two_press(self):
        self.filter_selected = 2
    def three_press(self):
        self.filter_selected = 3
    def four_press(self):
        self.filter_selected = 4
    def five_press(self):
        self.filter_selected = 5
    def rotate_background_image(self):
        self.background_image_index = self.background_image_index + 1
        if self.background_image_index > len(self.background_image_path_list):
            self.background_image_index = 0
    def change_kernel_size(self):
        self.kernel_size = self.kernel_size + 10
        if self.kernel_size > 100:
            self.kernel_size = 11

    def gstreamer_pipeline(self,sensor_id=0,capture_width=1280,capture_height=720,display_width=960,display_height=540,framerate=30,flip_method=0):
        """
        This function is used to create the gstreamer pipeline for the camera.
        Args:
            sensor_id (int, optional): The id of the camera. Defaults to 0.
            capture_width (int, optional): The width of the camera. Defaults to 1920.
            capture_height (int, optional): The height of the camera. Defaults to 1080.
            display_width (int, optional): The width of the display. Defaults to 1920.
            display_height (int, optional): The height of the display. Defaults to 1080.
            framerate (int, optional): The framerate of the camera. Defaults to 30.
            flip_method (int, optional): The flip method of the camera. Defaults to 0.
        Returns:
            str: The gstreamer pipeline.
        """
        return ('nvarguscamerasrc sensor-id =%d!'
                'video/x-raw(memory:NVMM), '
                'width=(int)%d, height=(int)%d, '
                'format=(string)NV12, framerate=(fraction)%d/1 ! '
                'nvvidconv flip-method=%d ! '
                'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink' %
                (sensor_id,
                 capture_width,
                 capture_height,
                 framerate,
                 flip_method,
                 display_width,
                 display_height)
                )
     
    def create_filter(self, frame, background_change = False):
        """This function will create the background blur and change the background to any image.
        Args:
            video_capture (_type_): _description_
        """
        result = self._model.infer(frame)
        result  = result.reshape((result.shape[1], result.shape[2], result.shape[0]))
        result = np.dstack((result, result, result))
        
        # print(result.shape)
        if background_change:
            final_frame = np.where(result == 0, self.background_image, frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return final_frame
        else:
            # Blur the captured frame
            # print(self.kernel_size)
            result_blur = cv.GaussianBlur(frame, (self.kernel_size, self.kernel_size),cv.BORDER_DEFAULT)
            final_frame = np.where(result == 0, result_blur, frame)
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            return final_frame
            
    def switch_model(self):
        if self.filter_selected == 1 or self.filter_selected == 2:
            self._model = self.pp_human_model
        if self.filter_selected == 3 or self.filter_selected == 4:
            print('Not implemented yet')
           

    def final_method(self):
        """
        This function is used to create the final method that is going to run all the CV windows.
        """
        try:
            self.window_title = 'Filter 1'
            # video_capture = cv.VideoCapture(self.gstreamer_pipeline(flip_method=0), cv.CAP_GSTREAMER)
            video_capture = cv.VideoCapture(0)
            if not video_capture.isOpened():
                print("Cannot open camera. Please check the camera connection and try again.")
                exit()
            else:
                # Set the camera resolution
                self.w = int(video_capture.get(cv.CAP_PROP_FRAME_WIDTH))
                self.h = int(video_capture.get(cv.CAP_PROP_FRAME_HEIGHT))
                print('Camera resolution: {}x{}'.format(self.w, self.h))
                # Set the tickmeter
                tm = cv.TickMeter()
                previous_filter = 1
                while True:

                    hasFrame, frame = video_capture.read()
                    if not hasFrame:
                        print('No frames grabbed!')
                        break
                    
                    # Handle all the key presses
                    key_pressed = cv.waitKey(1) & 0xFF
                    if key_pressed == 255:
                        pass
                    else: 
                        if key_pressed in self.key_mapper:
                            key_mapper = self.key_mapper[key_pressed]
                            if key_mapper == 'quit':
                                break
                            else:
                                key_mapper()

                    # Switch the model if the filter is changed
                    if previous_filter != self.filter_selected:
                        previous_filter = self.filter_selected
                        self.switch_model()
                        self.window_title = 'Filter ' + str(self.filter_selected)
                        cv.setWindowTitle(winname="Demo", title=self.window_title)
                        if self.filter_selected == 1:
                            self.func_name = self.create_filter
                            self.background_enable = False
                        elif self.filter_selected == 2:
                            self.func_name = self.create_filter
                            self.background_enable = True
                        elif self.filter_selected == 3:
                            result = self.create_filter(frame, background_change = False)
                        elif self.filter_selected == 4:
                            result = self.create_filter(frame, background_change = True)
                        elif self.filter_selected == 5:
                            pass
                        else:
                            pass
                    
                    partial_func = partial(self.func_name, frame)
                    tm.start()
                    result = partial_func(background_change = self.background_enable)
                    tm.stop()
                    
                    if self.enable_fps is not None:
                        cv.putText(result, 'FPS: {:.2f}'.format(tm.getFPS()), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))
                    cv.imshow("Demo", result)
                    tm.reset()

        except KeyboardInterrupt:
                print('Interrupted')

        finally:
            video_capture.release()
            cv.destroyAllWindows()


if __name__ == '__main__':
    filter_creator().final_method()