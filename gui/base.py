from tkinter import Canvas, Button, mainloop, Tk


class BaseGUI:
    def __init__(self, root=None, image_resolution=(480, 480)):
        self.root = root if root is not None else Tk()
        self.image_width, self.image_height = image_resolution
        self.left_image = Canvas(self.root, width=self.image_width, height=self.image_height)
        self.left_image.grid(row=0, column=0)
        self.right_image = Canvas(self.root, width=self.image_width, height=self.image_height)
        self.right_image.grid(row=0, column=2)
        self.next_set_button = Button(self.root, text='>>', command=self.next_set)
        self.next_set_button.grid(row=1, column=3)
        self.previous_set_button = Button(self.root, text='<<', command=self.previous_set)
        self.previous_set_button.grid(row=1, column=0)
        self.next_image_button = Button(self.root, text='>', command=self.next_image)
        self.next_image_button.grid(row=1, column=2)
        self.previous_image_button = Button(self.root, text='<', command=self.previous_image)
        self.previous_image_button.grid(row=1, column=1)
        self.quit_button = Button(self.root, text='CLOSE', command=self.quit)
        self.quit_button.grid(row=2, column=2)
        mainloop()

    def next_set(self):
        pass

    def next_image(self):
        pass

    def previous_set(self):
        pass

    def previous_image(self):
        pass

    def quit(self):
        exit()


BaseGUI()
