import tkinter
from tkinter import Canvas, Button, mainloop, Tk, filedialog

import cv2
from PIL import ImageTk, Image


from .pose_comparator import Comparator
from trainer_algorithm import Predictor
from prepare import predict


class BaseGUI:
    def __init__(self, root=None, image_resolution=(800, 60<0)):
        self.comparator = None
        self.sets = None
        self.index_of_set = None
        self.index_in_set = None
        self.root = root if root is not None else Tk()
        self.root.title("Baseball-Trainer")
        self.image_width, self.image_height = image_resolution
        self.model_2D_config = '../config/Model_2D.yaml'
        self.model_2D_weights = '../checkpoint/Model_2D.pkl'
        self.model_3D_weights = '../checkpoint/Model_3D.bin'
        self.left_image = None
        self.right_image = None
        self.left_canvas = Canvas(self.root, width=self.image_width, height=self.image_height)
        self.left_canvas.grid(row=0, column=0, columnspan=2)
        self.right_canvas = Canvas(self.root, width=self.image_width, height=self.image_height)
        self.right_canvas.grid(row=0, column=2, columnspan=2)
        self.next_set_button = Button(self.root, text='>>', command=self.next_set)
        self.next_set_button.grid(row=1, column=3)
        self.previous_set_button = Button(self.root, text='<<', command=self.previous_set)
        self.previous_set_button.grid(row=1, column=0)
        self.next_image_button = Button(self.root, text='>', command=self.next_image)
        self.next_image_button.grid(row=1, column=2)
        self.previous_image_button = Button(self.root, text='<', command=self.previous_image)
        self.previous_image_button.grid(row=1, column=1)
        self.quit_button = Button(self.root, text='CLOSE', command=self.quit)
        self.quit_button.grid(row=2, column=1, columnspan=2)
        self.good_pose_video = self.get_good_video_filename()
        self.bad_pose_video = self.get_bad_video_filename()
        self.init_comparator()
        mainloop()

    def file_dialog(self, filetypes, title):
        file_name = filedialog.askopenfilename(filetypes=filetypes,
                                               title=title)
        return file_name

    def get_bad_video_filename(self):
        return self.file_dialog(filetypes=[('.mp4files', '.mp4')], title='Select a video of you')

    def get_good_video_filename(self):
        return self.file_dialog(filetypes=[('.mp4files', '.mp4')], title='Select a video from a professional player')

    def update_images(self):

        if not self.index_of_set < len(self.sets):
            self.index_of_set = 0
            self.index_in_set = 0

        if self.index_in_set < 0:
            if self.index_of_set == 0:
                self.index_of_set = len(self.sets) - 1
            else:
                self.index_of_set -= 1
            self.index_in_set = len(self.sets[self.index_of_set]) - 1

        if not self.index_in_set < len(self.sets[self.index_of_set]):
            self.index_in_set = 0
            self.index_of_set += 1
        try:
            self.sets[self.index_of_set][self.index_in_set]
        except:
            self.update_images()
        left_image, right_image = self.sets[self.index_of_set][self.index_in_set]
        left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
        left_image = cv2.resize(left_image, (self.image_width, self.image_height))

        right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        right_image = cv2.resize(right_image, (self.image_width, self.image_height))

        self.left_image = ImageTk.PhotoImage(image=Image.fromarray(left_image))
        self.right_image = ImageTk.PhotoImage(image=Image.fromarray(right_image))

        self.left_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.left_image)
        self.right_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.right_image)

    def get_filename_from_path(self, path, with_extension=False):
        if with_extension:
            return path.split('/')[-1]
        return (path.split('/')[-1]).split('.')[0]

    def init_comparator(self):
        name_of_file_good_pose = self.get_filename_from_path(self.good_pose_video)
        name_of_file_bad_pose = self.get_filename_from_path(self.bad_pose_video)
        good_pose_prediction = "../predictions/" + name_of_file_good_pose+".npz"
        bad_pose_prediction = "../predictions/" + name_of_file_bad_pose +".npz"
        good_pose_metadata = "../predictions/" + name_of_file_good_pose + '_metadata'
        bad_pose_metadata = "../predictions/" + name_of_file_bad_pose + '_metadata'
        #predict(self.model_2D_config, self.model_2D_weights, self.bad_pose_video, bad_pose_prediction)
        #predict(self.model_2D_config, self.model_2D_weights, self.good_pose_video, good_pose_prediction)
        bad_pose_data = "./data/data_2d_custom_" + name_of_file_bad_pose
        good_pose_data = "./data/data_2d_custom_" + name_of_file_good_pose
        #Predictor(bad_pose_prediction, self.model_3D_weights, export_path=bad_pose_prediction).export_prediction()
        #Predictor(good_pose_prediction, self.model_3D_weights, export_path=good_pose_prediction).export_prediction()

        comp = Comparator(good_pose='../predictions/baseball_1.npy',
                          bad_pose='../predictions/baseball_me_2.npy',
                          good_pose_video='../inputs/baseball_1.mp4',
                          bad_pose_video='../inputs/me_2.mp4',
                          good_bboxes='../predictions/baseball_1_metadata.npy',
                          bad_bboxes='../predictions/baseball_me_2_metadata.npy')
        #comp = Comparator(good_pose_prediction, bad_pose_prediction, good_pose_metadata, bad_pose_metadata,self.good_pose_video, self.bad_pose_video)
        self.sets = comp.compare_poses()
        self.index_in_set = 0
        self.index_of_set = 0
        self.update_images()

    def next_set(self):
        self.index_of_set += 1
        self.index_in_set = 0
        self.update_images()

    def next_image(self):
        self.index_in_set += 1
        self.update_images()

    def previous_set(self):
        self.index_of_set -= 1
        self.index_in_set = 0
        self.update_images()

    def previous_image(self):
        self.index_in_set -= 1
        self.update_images()

    def quit(self):
        exit()


BaseGUI()
