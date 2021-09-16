import os

import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk

def marker_watershed(img, smoothing, starting_markers = None):
    img = cv2.GaussianBlur(img, (smoothing,smoothing), 0)
    img_thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, smoothing, -2)

    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    k7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

    sure_fg = cv2.erode(img_thresh, k3, iterations = 1)

    if starting_markers is None:
        nmarkers, markers = cv2.connectedComponents(sure_fg)
    else:
        markers = starting_markers
        nmarkers = int(np.max(starting_markers)) + 1

    sure_bg = cv2.dilate(sure_fg, k7, iterations = 5)
    contours,_ = cv2.findContours(sure_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(markers, contours, -1, nmarkers, 1)
    markers = markers.astype('int32')

    img = cv2.cvtColor(np.zeros(img.shape, 'uint8'), cv2.COLOR_GRAY2RGB)
    cv2.watershed(img, markers)
    markers[np.logical_or(markers == -1, markers == nmarkers)] = 0

    return markers

def calc_voronoi(size, peaks):
    subdiv = cv2.Subdiv2D((0,0,size[0],size[1]))
    subdiv.insert(peaks)

    facets, centers = subdiv.getVoronoiFacetList([])

    grid_points = [(x,y) for x in range(size[0]) for y in range(size[1])]
    ret = map(subdiv.findNearest, grid_points)
    grid = np.asarray([t[0] for t in ret])
    grid = np.reshape(grid, size)
    return grid

class Point():
    sz = 2

    def __init__(self, canvas, coords, number, show_lab):
        self.canvas = canvas
        self.coords = coords
        self.number = number
        lab_state = tk.NORMAL if show_lab else tk.HIDDEN

        x = coords[0]
        y = coords[1]
        self.pt = self.canvas.create_oval(x-Point.sz,y-Point.sz,x+Point.sz,y+Point.sz, fill = '#000099', tags = "point")
        self.lab = self.canvas.create_text(coords, anchor = "nw", text = str(self.number), font = ("Purisa", 7), state = lab_state)

    def move(self, dx, dy, absolute = False):
        if absolute:
            self.coords = (dx,dy)
            self.canvas.coords(self.lab, (dx,dy))
            self.canvas.coords(self.pt, (dx-self.sz,dy-self.sz,dx+self.sz,dy+self.sz))
        else:
            self.coords = (self.coords[0] + dx, self.coords[1] + dy)
            self.canvas.move(self.lab, dx, dy)
            self.canvas.move(self.pt, dx, dy)

    def erase(self):
        self.canvas.delete(self.lab)
        self.canvas.delete(self.pt)

    def renumber(self, number):
        self.number = number
        self.canvas.itemconfig(self.lab, text = str(number))

    def set_active(self, is_active):
        self.canvas.itemconfig(self.pt, fill = '#00ff00' if is_active else '#0000ff')

    def set_lab_state(self, is_normal):
        self.canvas.itemconfig(self.lab, state = tk.NORMAL if is_normal else tk.HIDDEN)

class App(tk.Frame):
    def __init__(self):
        self.root = tk.Tk()
        super().__init__(self.root)
        self.mouse_moved = None
        self.mouse_starting = ()
        self.selection_rect = None

        self.flood_image = None
        self.canvas_image = None
        self.show_labs_state = tk.IntVar()
        self.show_grid_state = tk.IntVar()
        self.ncols = 19
        self.smoothing = 11

        self.points = []
        self.curr_point = []
        self.canvas_size = ()
        self.curr_file = tk.StringVar()
        self.curr_file.set("No file selected")
        self.draw()

        self.last_dir = None

    def toggle_labs(self):
        lab_state = self.show_labs_state.get()
        [p.set_lab_state(lab_state) for p in self.points]

    def toggle_grid(self):
        grid_state = tk.NORMAL if self.show_grid_state.get() else tk.HIDDEN
        [self.flood.itemconfig(g, state = grid_state) for g in self.flood.find_withtag("grid")]

    def clear_points(self, pts):
        [p.erase() for p in pts]

        self.points     = [p for p in self.points     if p not in pts]
        self.curr_point = [p for p in self.curr_point if p not in pts]

        self.flood.itemconfigure(self.xtal_counter, text = str(len(self.points)))

    def points_in_box(self, bbox):
        pts = self.flood.find_overlapping(*bbox)
        pts = [p for p in pts if 'point' in self.flood.gettags(p)]
        return [p for p in self.points if p.pt in pts]

    def get_neighbor(self, pt, right_neighbor = True):

        if right_neighbor:
            w = 0.2  * self.canvas_size[0]
            h = 0.05 * self.canvas_size[1]
            bbox = (pt.coords[0], pt.coords[1] - h, pt.coords[0] + w, pt.coords[1] + h)
        else:
            w = 0.05 * self.canvas_size[0]
            h = 0.2  * self.canvas_size[1]
            bbox = (pt.coords[0] - w, pt.coords[1], pt.coords[0] + w, pt.coords[1] + h)

        neighbors = self.points_in_box(bbox)
        coords = np.array([n.coords for n in neighbors])

        # If searching for neighbor below, flip coordinates
        if right_neighbor:
            pt_coords = np.array(pt.coords)
        else:
            pt_coords = np.flip(np.array(pt.coords))
            coords = np.flip(coords, 1)

        disp = coords - pt_coords # normalize to canvas size?

        neighbors = [n for ((a,b),n) in zip(disp,neighbors) if a > 0]
        disp = [(a,b) for (a,b) in disp if a > 0]

        if len(neighbors) == 0: return None

        distance = np.linalg.norm(disp, 2, 1)

        pt_tan = [b/a for (a,b) in disp]
        pt_angle = np.abs(np.arctan(pt_tan))

        weighting = lambda dst,phi,a,b: a*dst + b*phi
        pt_weight = [weighting(dst,phi,5,100) for (dst,phi) in zip(distance,pt_angle)]

        return neighbors[np.argmin(pt_weight)]

    def find_peaks(self):
        if self.flood_image is None:
            return

        self.flood.delete('grid')
        self.clear_points(self.points)

        regions = marker_watershed(self.flood_image, self.smoothing)
        ref_img = cv2.GaussianBlur(self.flood_image, (self.smoothing,self.smoothing), 0)

        npoints = 0
        locs = []

        for r in np.unique(regions):
            if (r > 0):
                mask = np.zeros(regions.shape, 'uint8')
                mask[regions == r] = 1
                _,_,_,p = cv2.minMaxLoc(ref_img, mask)

                x = p[0] * (self.canvas_size[0] / self.flood_image.shape[0])
                y = p[1] * (self.canvas_size[1] / self.flood_image.shape[1])

                nearby = self.flood.find_overlapping(x-2,y-2,x+2,y+2) 
                if not any([ ('point' in self.flood.gettags(p)) for p in nearby ]):
                    self.points.append(Point(self.flood, (x,y), -1, self.show_labs_state.get()))

        self.flood.itemconfigure(self.xtal_counter, text = str(len(self.points)))

    def order_peaks(self):
        self.flood.delete('grid')

        [p.renumber(-1) for p in self.points]
        locs = [p.coords for p in self.points]
        npoints = len(self.points)

        # Pick the crystal closest to the top left corner as #1
        dst = np.linalg.norm(locs, 2, 1)
        first_xtal = np.argmin(dst)

        idx = [-1] * npoints

        for i in range(npoints):
            if i == 0:
                this_point_idx = first_xtal
            else:
                if i % self.ncols == 0:
                    # Just finished a row, move to next
                    last_point = self.points[idx[i - self.ncols]]
                    this_point = self.get_neighbor(last_point, False)
                else:
                    # Find next point in row
                    last_point = self.points[idx[i - 1]]
                    this_point = self.get_neighbor(last_point, True)

                if this_point == None: continue

                this_point_idx = self.points.index(this_point)

            if self.points[this_point_idx].number == -1:
                idx[i] = this_point_idx
                self.points[this_point_idx].renumber(i + 1)

        grid_state = tk.NORMAL if self.show_grid_state.get() else tk.HIDDEN

        for i in range(npoints): 
            p = self.points[idx[i]]

            if idx[i] is not -1:
                p_r = self.points[idx[i + 1]]           if (i + 1) % self.ncols > 0  else None # index out of range error here
                p_b = self.points[idx[i + self.ncols]]  if i + self.ncols < npoints  else None

                if p_r is not None and p_r.number is not -1:
                    self.flood.create_line(p.coords + p_r.coords, fill = 'grey', tags = 'grid', state = grid_state)

                if p_b is not None and p_b.number is not -1:
                    self.flood.create_line(p.coords + p_b.coords, fill = 'grey', tags = 'grid', state = grid_state)

        self.curr_point = [p for p in self.points if p.number is -1]
        [p.set_active(True) for p in self.curr_point]

    def save_peaks(self):
        x_ratio = self.flood_image.shape[0] / self.canvas_size[0]
        y_ratio = self.flood_image.shape[1] / self.canvas_size[1]

        point_values  = [p.number for p in self.points]
        scaled_points = [(round(p.coords[0]*x_ratio), round(p.coords[1]*y_ratio)) for p in self.points]

        voronoi = calc_voronoi(self.flood_image.shape, scaled_points)
        regions = np.zeros(voronoi.shape, dtype = 'int32')

        for loc, val in zip(scaled_points, point_values):
            regions[np.equal(voronoi,voronoi[loc])] = val

        regions = np.transpose(regions)
        regions = np.flipud(regions)

        file_base_name = os.path.normpath(self.curr_file.get())
        file_base_name = os.path.basename(file_base_name)
        file_base_name = os.path.splitext(file_base_name)[0]

        initialdir = self.last_dir or os.getcwd()

        fname = asksaveasfilename(
                initialdir = initialdir,
                initialfile = file_base_name,
                title = "Save segmented data", 
                defaultextension = ".clu",
                filetypes = (("Crystal lookup table","*.clu"), ("All FIles","*.*"))
                )

        self.last_dir = os.path.dirname(fname)

        try:
            regions.tofile(fname)
        except Exception as e:
            print(str(e))

    def load_peaks(self):
        pass

    def load_file(self):
        self.flood.delete('grid')
        self.clear_points(self.points)

        if self.canvas_image is not None:
            self.flood.delete(self.canvas_image)
            self.flood_image = None

        initialdir = self.last_dir or os.getcwd()
        fname = askopenfilename(
                initialdir = initialdir, 
                title = "Select file",
                defaultextension = ".raw",
                filetypes = (("Raw data","*.raw"), ("All FIles","*.*"))
                )
        self.last_dir = os.path.dirname(fname)

        self.curr_file.set(fname)
        self.root.update()

        self.canvas_size = (self.flood.winfo_width(), self.flood.winfo_height())

        try:
            self.flood_image = np.fromfile(fname, 'int32').astype('uint8').reshape((512,512))
            self.flood_image = np.flipud(self.flood_image)

            self.flood_image = cv2.GaussianBlur(self.flood_image, (5,5), 0)
            self.flood_image = cv2.Laplacian(self.flood_image , cv2.CV_64F, ksize = 7)

            self.flood_image = self.flood_image / np.min(self.flood_image)
            self.flood_image[self.flood_image < 0] = 0

            cm = plt.get_cmap('magma_r')
            self.flood_image_raw = Image.fromarray(cm(self.flood_image, bytes = True))

            self.flood_image = self.flood_image * 255
            self.flood_image = self.flood_image.astype('uint8')

            self.flood_image_scale = ImageTk.PhotoImage(image = self.flood_image_raw.resize(self.canvas_size, Image.ANTIALIAS))
            self.canvas_image = self.flood.create_image(0, 0, anchor = tk.NW, image = self.flood_image_scale)
            self.flood.tag_lower(self.canvas_image)

        except Exception as e:
            print(str(e))
            self.flood_image = None
            self.curr_file.set("No file selected")

    def quit(self, ev = None):
        self.root.destroy()

    def image_rescale(self):
        if self.flood_image is not None:
            self.flood_image_scale = ImageTk.PhotoImage(image = self.flood_image_raw.resize(self.canvas_size, Image.ANTIALIAS))
            self.flood.itemconfig(self.canvas_image, image = self.flood_image_scale)

    def window_resize(self, ev):
        self.flood.delete('grid')

        if self.flood_image is not None:
            self.image_rescale()

            for p in self.points:
                x = ev.width  * p.coords[0] / self.canvas_size[0]
                y = ev.height * p.coords[1] / self.canvas_size[1]
                p.move(x, y, True)
            
        self.canvas_size = (ev.width, ev.height)

    def mouse_lpress(self, ev):
        self.mouse_moved = False
        self.mouse_starting = (ev.x, ev.y)
        self.selection_rect = self.flood.create_rectangle(ev.x,ev.y,ev.x,ev.y, outline = '#ff0000')

    def mouse_lrelease(self, ev):
        self.flood.delete(self.selection_rect)

        if self.mouse_moved:
            # Identify points in the bounding box
            self.curr_point = self.points_in_box((self.mouse_starting[0], self.mouse_starting[1], ev.x, ev.y))
        else:
            # Find points near the click event
            pts = self.points_in_box((ev.x-Point.sz,ev.y-Point.sz,ev.x+Point.sz,ev.y+Point.sz))

            if len(pts) > 0:
                # Find the nearest point to the mouse location
                p_coords = [p.coords for p in pts]
                a = np.array([(p[0]+Point.sz, p[1]+Point.sz) for p in p_coords])
                b = np.array((ev.x,ev.y))
                dst = np.linalg.norm(a-b, 2, 1)
                self.curr_point = [pts[np.argmin(dst)]]
            else:
                # Create a new point
                new_pt = Point(self.flood, (ev.x,ev.y), len(self.points)+1, self.show_labs_state.get())
                self.points.append(new_pt)
                self.curr_point = [new_pt]
                self.flood.itemconfigure(self.xtal_counter, text = str(len(self.points)))

        for p in self.points:
            p.set_active(p in self.curr_point)

    def mouse_lmotion(self, ev):
        self.mouse_moved = True
        self.flood.coords(self.selection_rect, self.mouse_starting[0], self.mouse_starting[1], ev.x, ev.y)

    def mouse_rpress(self, ev):
        self.clear_points(self.curr_point)

    def nudge_point(self, dx, dy):
        [p.move(dx, dy) for p in self.curr_point]

    def draw(self):
        self.file_name = tk.Label(self.root, textvariable = self.curr_file)
        self.flood = tk.Canvas(self.root, bd = 1, highlightthickness = 0, relief = 'ridge')
        self.xtal_counter = self.flood.create_text((5,5), anchor = "nw", text = str(len(self.points)))

        self.file_button = tk.Button(self.root, text = "File", command = self.load_file)
        self.seg_button  = tk.Button(self.root, text = "Find Peaks", command = self.find_peaks)
        self.ord_button  = tk.Button(self.root, text = "Order Peaks", command = self.order_peaks)
        self.vor_button  = tk.Button(self.root, text = "Save Peaks", command = self.save_peaks)
        self.quit_button = tk.Button(self.root, text = "Quit", command = self.quit)
        self.show_labs   = tk.Checkbutton(self.root, text = "Show labels", variable = self.show_labs_state, command = self.toggle_labs)
        self.show_grid   = tk.Checkbutton(self.root, text = "Show grid", variable = self.show_grid_state, command = self.toggle_grid)

        self.file_name.pack(fill = tk.X, padx = 5, pady = 5)
        self.flood.pack(fill = tk.BOTH, expand = 1, padx = 5, pady = 5)
        self.file_button.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.seg_button.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.ord_button.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.vor_button.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.quit_button.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.show_labs.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)
        self.show_grid.pack(side = tk.LEFT, fill = tk.X, padx = 5, pady = 5)

        self.flood.bind('<ButtonPress-1>', self.mouse_lpress)
        self.flood.bind('<ButtonRelease-1>', self.mouse_lrelease)
        self.flood.bind('<ButtonPress-3>', self.mouse_rpress)
        self.flood.bind('<B1-Motion>', self.mouse_lmotion)
        self.flood.bind('<Configure>', self.window_resize)

        #self.root.bind('<Up>',    lambda e: self.nudge_point( 0,-1))
        #self.root.bind('<Down>',  lambda e: self.nudge_point( 0, 1))
        #self.root.bind('<Left>',  lambda e: self.nudge_point(-1, 0))
        #self.root.bind('<Right>', lambda e: self.nudge_point( 1, 0))

        self.root.bind('w', lambda e: self.nudge_point( 0,-1))
        self.root.bind('s', lambda e: self.nudge_point( 0, 1))
        self.root.bind('a', lambda e: self.nudge_point(-1, 0))
        self.root.bind('d', lambda e: self.nudge_point( 1, 0))

        self.root.bind('<Escape>', self.quit)

        self.canvas_size = (self.flood.winfo_width(), self.flood.winfo_height())

app = App()
app.mainloop()
