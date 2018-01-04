#! /usr/bin/env python3

# Ellipse editor, standalone utility for use with steelpan images & annotations
# Author: Scott H. Hawley     no license yet.

#TODO:
# - Add text file read option
# - Add text file save option
# - Add keeping track of # of rings


import tkinter as tk
import math
import numpy as np
from PIL import Image, ImageTk
import tkinter.font
import argparse
import os
import glob

def poly_oval(xc, yc, a, b, angle=0, steps=100 ):
    # From https://mail.python.org/pipermail/python-list/2000-December/022013.html
    """return an oval as coordinates suitable for create_polygon"""

    # angle is in degrees anti-clockwise, convert to radians
    rotation = angle * math.pi / 180.0

    point_list = []

    # create the oval as a list of points
    for i in range(steps):

        # Calculate the angle for this step
        # 360 degrees == 2 pi radians
        theta = (math.pi * 2) * (float(i) / steps)

        x1 = a * math.cos(theta)
        y1 = b * math.sin(theta)

        # rotate x, y
        x = (x1 * math.cos(rotation)) + (y1 * math.sin(rotation))
        y = (y1 * math.cos(rotation)) - (x1 * math.sin(rotation))

        point_list.append((x + xc))
        point_list.append((y + yc))

    return point_list



class EllipseEditor(tk.Frame):
    '''Edit ellipses for steelpan images'''

    def __init__(self, parent, img_file, txt_file):
        tk.Frame.__init__(self, parent)

        # create a canvas
        self.width, self.height = 512, 384
        self.readout = 300
        self.canvas = tk.Canvas(width=self.width + self.readout, height=self.height )
        self.canvas.pack(fill="both", expand=True)
        self.img_file = img_file
        self.txt_file = txt_file

        self.color = "green"

        image = Image.open(img_file)
        tkimage = ImageTk.PhotoImage(image=image)
        label = tk.Label(image=tkimage)
        label.image = tkimage # keep a reference!
        #label.pack()
        self.canvas.create_image(self.width/2,self.height/2, image=tkimage)

        # this data is used to keep track of an
        # item being dragged
        self._drag_data = {"x": 0, "y": 0, "items": None}

        self._token_data = []
        self._numtokens = 0
        self.hr = 3             # handle radius

        # global bindings
        self.canvas.tag_bind(tk.ALL, "<B1-Motion>", self.update_readout)
        self.canvas.tag_bind(tk.ALL, "<Double-Button-1>", self.on_doubleclick)


        self.infostr = self.img_file
        self.text = self.canvas.create_text(self.width+10, 10, text=self.infostr, anchor=tk.NW, font=tk.font.Font(size=16))

        # create some ellipse tokens (and their handles)
        n_obj = 6
        for i in range(n_obj):
            xc, yc = int(np.random.rand()*self.width), int(np.random.rand()*self.height)
            a, b = int(np.random.rand()*self.width/3), int(np.random.rand()*self.height/3)
            angle = int(np.random.rand()*180)
            self._create_token((xc, yc), (a, b), angle, self.color)
        self.update_readout(None)



    def _create_token(self, coord, axes, angle, color):
        '''Create a token at the given coordinate in the given color'''
        self._numtokens += 1
        (x,y) = coord
        (a,b) = axes
        #self.canvas.create_oval(x-a, y-b, x+a, y+b, outline=color, fill=None, width=3, tags="token")
        thistag = "token"+str(self._numtokens)
        oval = self.canvas.create_polygon(*tuple(poly_oval(x, y, a, b, angle=angle)),outline=color, fill='', width=3, tags=(thistag,"main"))

        # handles for resize / rotation
        h_a_x, h_a_y = x + a*np.cos(np.deg2rad(angle)),  y - a*np.sin(np.deg2rad(angle))
        h_b_x, h_b_y = x + b*np.sin(np.deg2rad(angle)),  y + b*np.cos(np.deg2rad(angle))
        h_a = self.canvas.create_oval(h_a_x-self.hr, h_a_y-self.hr, h_a_x+self.hr, h_a_y+self.hr, outline=color, fill=color, width=3, tags=(thistag,"handle","axis_a"))
        h_b = self.canvas.create_oval(h_b_x-self.hr, h_b_y-self.hr, h_b_x+self.hr, h_b_y+self.hr, outline=color, fill="blue", width=3, tags=(thistag,"handle","axis_b"))

        self._token_data.append([oval,h_a,h_b])

        self.canvas.tag_bind("main", "<ButtonPress-1>", self.on_main_press)
        self.canvas.tag_bind("main", "<ButtonRelease-1>", self.on_main_release)
        self.canvas.tag_bind("main", "<B1-Motion>", self.on_main_motion)

        self.canvas.tag_bind("handle", "<ButtonPress-1>", self.on_handle_press)
        self.canvas.tag_bind("handle", "<ButtonRelease-1>", self.on_handle_release)
        self.canvas.tag_bind("handle", "<B1-Motion>", self.on_handle_motion)


    def on_main_press(self, event):
        '''Begining drag of an object'''
        # record the item and its location
        obj_id = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags( obj_id )
        #print("Main:  ids, tags = ",ids, tags)
        self._drag_data["items"] = tags[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_main_release(self, event):
        '''End drag of an object'''
        # if object is off the screen, delete it
        if ((event.x < 0) or (event.y < 0 ) or (event.x > self.width) or (event.y > self.height)):
            self.canvas.delete(self._drag_data["items"])
            self.update_readout(None)
        # reset the drag information
        self._drag_data["items"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def on_main_motion(self, event):
        '''Handle dragging of an object'''
        # compute how much the mouse has moved
        delta_x = event.x - self._drag_data["x"]
        delta_y = event.y - self._drag_data["y"]
        # move the object the appropriate amount
        self.canvas.move(self._drag_data["items"], delta_x, delta_y)
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y



    def retrieve_ellipse_info(self, tokentag):
        tokenitems = self.canvas.find_withtag( tokentag )
        [main_id, axis_a_id, axis_b_id ]= tokenitems

        ell_coords = self.canvas.coords(main_id)   # coordinates of all points in ellipse
        xcoords, ycoords = ell_coords[0::2],  ell_coords[1::2]
        xc, yc = np.mean( xcoords ),  np.mean( ycoords )   # coordinates of center of ellipse

        h_a_coords = self.canvas.coords(axis_a_id)
        h_a_x, h_a_y = np.mean( h_a_coords[0::2] ),  np.mean( h_a_coords[1::2] )
        a = np.sqrt( (h_a_x - xc)**2 + (h_a_y - yc)**2 )

        h_b_coords = self.canvas.coords(axis_b_id)
        h_b_x, h_b_y = np.mean( h_b_coords[0::2] ),  np.mean( h_b_coords[1::2] )
        b = np.sqrt( (h_b_x - xc)**2 + (h_b_y - yc)**2 )

        angle = np.rad2deg( np.arctan2( yc - h_a_y, h_a_x - xc) )

        return xc, yc, a, b, angle, ell_coords



    def on_handle_press(self, event):
        '''Begining drag of an handle'''
        # record the item and its location
        ids = self.canvas.find_closest(event.x, event.y)[0]
        tags = self.canvas.gettags( ids )
        self._drag_data["items"] = self.canvas.find_closest(event.x, event.y)[0]
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_handle_release(self, event):
        '''End drag of an handle'''
        # reset the drag information
        self._drag_data["items"] = None
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0

    def on_handle_motion(self, event):
        '''Handle dragging of an handle'''
        # compute how much the mouse has moved
        oldx, oldy = self._drag_data["x"], self._drag_data["y"]
        delta_x = event.x - oldx
        delta_y = event.y - oldy
        # move the handle the appropriate amount
        self.canvas.move(self._drag_data["items"], delta_x, delta_y)

        # what are the tags for this particular handle
        tags = self.canvas.gettags( self._drag_data["items"] )
        tokentag = tags[0]
        xc, yc, a, b, angle, coords  = self.retrieve_ellipse_info( tokentag )
        #print(" Hey: xc, yc, a, b, angle = ",xc, yc, a, b, angle)

        tokenitems = self.canvas.find_withtag( tokentag )
        [main_id, axis_a_id, axis_b_id ]= tokenitems

        new_r = np.sqrt( (event.x -xc)**2 + (event.y - yc)**2 )
        new_angle = np.rad2deg( np.arctan2( yc-oldy, oldx-xc) )

        # which handle is currently being manipulated?
        if ("axis_a" in tags):
            b_coords = self.canvas.coords(axis_b_id)
            h_b_x, h_b_y = np.mean( b_coords[0::2] ),  np.mean( b_coords[1::2] )
            new_coords = poly_oval( xc, yc, new_r, b, angle=new_angle)
            h_b_x, h_b_y =  xc + b*np.sin(np.deg2rad(new_angle)),  yc + b*np.cos(np.deg2rad(new_angle))
            self.canvas.coords(axis_b_id, [ h_b_x-self.hr, h_b_y-self.hr, h_b_x+self.hr, h_b_y+self.hr] )
        elif ("axis_b" in tags):
            a_coords = self.canvas.coords(axis_a_id)
            h_a_x, h_a_y = np.mean( a_coords[0::2] ),  np.mean( a_coords[1::2] )
            new_angle = new_angle + 90   # a and b axes are offset by 90 degrees; angle is defined relative to a axis
            new_coords = poly_oval( xc, yc, a, new_r, angle=new_angle)
            h_a_x, h_a_y =  xc + a*np.cos(np.deg2rad(new_angle)),  yc - a*np.sin(np.deg2rad(new_angle))
            self.canvas.coords(axis_a_id, [ h_a_x-self.hr, h_a_y-self.hr, h_a_x+self.hr, h_a_y+self.hr] )
        else:
            print("Error: bad tags")
            assert(0==1)

        self.canvas.coords(main_id, new_coords)  # reassign coords for the entire ellipse (i.e. 'redraw')
        # record the new position
        self._drag_data["x"] = event.x
        self._drag_data["y"] = event.y

    def on_doubleclick(self, event):  # create a new ellipse
        xc, yc = event.x, event.y
        a, b = 50, 50
        angle = 0
        self._create_token((xc, yc), (a, b), angle, self.color)
        self.update_readout(None)

    def update_readout(self, event):
        mains = self.canvas.find_withtag( "main" )
        self.infostr = self.img_file+':\n'
        for main_id in mains:
            tokentag = self.canvas.gettags( main_id )[0]
            xc, yc, a, b, angle, coords = self.retrieve_ellipse_info( tokentag )
            self.infostr += '[{:4d}, {:4d}, {:4d}, {:4d}, {:6.2f}]\n'.format(int(xc), int(yc), int(a), int(b), angle)
        self.canvas.itemconfigure(self.text, text=self.infostr)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Edit a file set of files.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #parser.add_argument('file', nargs='+', help='Path of a file or a list of files.')
    parser.add_argument('file', nargs='+', help='Path of a file or a list of files.')
    args = parser.parse_args()
    files = list(args.file)
    print("files = ",files)

    print("Instructions:")
    print("- Double-click to create ellipse")
    print("- Click and drag to move ellipse")
    print("- Click and drag 'handles' to resize/rotate ellipse (solid = 'a', hollow = 'b')")
    print("- Drag off-screen to destroy/delete ellipse")

    root = tk.Tk()
    EllipseEditor(root, 'test_img.png', 'test_img.txt').pack(fill="both", expand=True)
    root.mainloop()
