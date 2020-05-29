import tkinter
from tkinter import Canvas, Button, mainloop, Tk

from PIL import ImageTk, Image

from pose_comparator import Comparator


class BaseGUI:
    def __init__(self, root=None, image_resolution=(800, 600)):
        self.comparator = None
        self.sets = None
        self.index_of_set = None
        self.index_in_set = None
        self.root = root if root is not None else Tk()
        self.image_width, self.image_height = image_resolution
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
        self.quit_button.grid(row=2, column=1, columnspan=2j)
        self.init_comparator()
        mainloop()

    def update_images(self):
        if not self.index_in_set < len(self.sets[self.index_of_set]):
            self.index_in_set = 0
            self.index_of_set += 1
        if self.index_in_set < 0:
            if self.index_of_set == 0:
                self.index_of_set = len(self.sets) - 1
            else:
                self.index_of_set -= 1
            self.index_in_set = len(self.sets[self.index_of_set]) - 1

        if not self.index_of_set < len(self.sets):
            self.index_of_set = 0
            self.index_in_set = 0

        left_image, right_image = self.sets[self.index_of_set][self.index_in_set]
        self.left_image = ImageTk.PhotoImage(image=Image.fromarray(left_image))
        self.right_image = ImageTk.PhotoImage(image=Image.fromarray(right_image))
        self.left_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.left_image)
        self.right_canvas.create_image(0, 0, anchor=tkinter.NW, image=self.right_image)

    def init_comparator(self):
        comp = Comparator(good_pose='../predictions/baseball_1.npy',
                          bad_pose='../predictions/baseball_me_2.npy',
                          good_pose_video='../inputs/baseball_1.mp4',
                          bad_pose_video='../inputs/me_2.mp4',
                          good_bboxes='../predictions/baseball_1_metadata.npy',
                          bad_bboxes='../predictions/baseball_me_2_metadata.npy')
        self.sets = comp.compare_poses()
        self.index_in_set = 0
        self.index_of_set = 0
        self.update_images()

    def next_set(self):
        self.index_of_set += 1
        self.update_images()

    def next_image(self):
        self.index_in_set += 1
        self.update_images()

    def previous_set(self):
        self.index_of_set -= 1
        self.update_images()

    def previous_image(self):
        self.index_in_set += 1
        self.update_images()

    def quit(self):
        exit()


BaseGUI()
