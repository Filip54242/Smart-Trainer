import os
import tkinter
from tkinter import Canvas, Button, mainloop, Tk, filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
from PIL import ImageTk, Image

from pose_comparator import Comparator
from prepare import predict
from trainer_algorithm import Predictor


class BaseGUI:
    def __init__(self, root=None, image_resolution=(800, 600), enable_cuda=False):
        self.with_cuda = enable_cuda
        self.comparator = None
        self.sets = None
        self.index_of_set = None
        self.index_in_set = None
        self.root = root if root is not None else Tk()
        self.root.title("Baseball-Trainer")
        self.image_width, self.image_height = image_resolution
        self.model_2D_config = './config/Model_2D.yaml'
        self.model_2D_weights = './checkpoint/Model_2D.pkl'
        self.model_3D_weights = './checkpoint/Model_3D.bin'
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
        self.quit_button.grid(row=3, column=1, columnspan=2)
        self.choose_again_button = Button(self.root, text='CHOOSE AGAIN', command=self.init_comparator)
        self.choose_again_button.grid(row=2, column=1, columnspan=2)
        self.bad_pose_video = None
        self.good_pose_video = None
        self.loading_photo = './aux/loading.jpg'
        self.get_loading_image()
        self.init_comparator()
        mainloop()

    def file_dialog(self, filetypes, title):
        file_name = filedialog.askopenfilename(filetypes=filetypes,
                                               title=title)
        return file_name

    def get_loading_image(self):
        self.loading_photo = plt.imread(self.loading_photo)
        self.loading_photo = cv2.cvtColor(self.loading_photo, cv2.COLOR_BGR2RGB)
        self.loading_photo = cv2.resize(self.loading_photo, (self.image_width, self.image_height))
        self.loading_photo = ImageTk.PhotoImage(image=Image.fromarray(self.loading_photo))

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

    def loading(self):
        self.left_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.loading_photo)
        self.right_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.loading_photo)

    def get_video_paths(self):
        self.good_pose_video = self.get_good_video_filename()
        if type(self.good_pose_video) is not str or not os.path.isfile(self.good_pose_video):
            messagebox.showerror("Error", "You can't continue if you don't select a video of a professional player")
            return False

        self.bad_pose_video = self.get_bad_video_filename()
        if type(self.bad_pose_video) is not str or not os.path.isfile(self.bad_pose_video):
            messagebox.showerror("Error", "You can't continue if you don't select a video of you")
            return False
        return True

    def init_comparator(self):
        self.loading()
        if self.get_video_paths():
            name_of_file_good_pose = self.get_filename_from_path(self.good_pose_video)
            name_of_file_bad_pose = self.get_filename_from_path(self.bad_pose_video)

            good_pose_prediction = "./predictions/" + name_of_file_good_pose
            bad_pose_prediction = "./predictions/" + name_of_file_bad_pose

            good_pose_metadata = "./predictions/" + name_of_file_good_pose + '_metadata.npy'
            bad_pose_metadata = "./predictions/" + name_of_file_bad_pose + '_metadata.npy'

            if not os.path.isfile(bad_pose_prediction + '.npz'):
                predict(self.model_2D_config, self.model_2D_weights, self.bad_pose_video, bad_pose_prediction,
                        with_cuda=self.with_cuda)
            if not os.path.isfile(good_pose_prediction + '.npz'):
                predict(self.model_2D_config, self.model_2D_weights, self.good_pose_video, good_pose_prediction,
                        with_cuda=self.with_cuda)

            good_pose_prediction += '.npz'
            bad_pose_prediction += '.npz'

            good_final_data = './predictions/' + name_of_file_good_pose + '.npy'
            bad_final_data = './predictions/' + name_of_file_bad_pose + '.npy'

            Predictor(bad_pose_prediction, self.model_3D_weights, export_path=bad_final_data,
                      with_cude=self.with_cuda).export_prediction()
            Predictor(good_pose_prediction, self.model_3D_weights, export_path=good_final_data,
                      with_cude=self.with_cuda).export_prediction()

            comp = Comparator(good_pose=good_final_data,
                              bad_pose=bad_final_data,
                              good_bboxes=good_pose_metadata,
                              bad_bboxes=bad_pose_metadata,
                              good_pose_video=self.good_pose_video,
                              bad_pose_video=self.bad_pose_video)
            self.sets = comp.compare_poses()
            if len(self.sets) == 0:
                messagebox.showerror("Error", "There is nothing to show")
            else:
                self.index_in_set = 0
                self.index_of_set = 0
                self.update_images()

    def next_set(self):
        if self.sets is not None and len(self.sets) != 0:
            self.index_of_set += 1
            self.index_in_set = 0
            self.update_images()

    def next_image(self):
        if self.sets is not None and len(self.sets) != 0:
            self.index_in_set += 1
            self.update_images()

    def previous_set(self):
        if self.sets is not None and len(self.sets) != 0:
            self.index_of_set -= 1
            self.index_in_set = 0
            self.update_images()

    def previous_image(self):
        if self.sets is not None and len(self.sets) != 0:
            self.index_in_set -= 1
            self.update_images()

    def quit(self):
        exit()
