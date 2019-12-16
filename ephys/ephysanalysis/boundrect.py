from __future__ import division
from __future__ import print_function
import sys
import numpy as np
import math
import matplotlib.pyplot as mpl
import scipy.spatial

# Find the minimum-area bounding box of a set of 2D points
#
# The input is a 2D convex hull, in an Nx2 numpy array of x-y co-ordinates.
# The first and last points points must be the same, making a closed polygon.
# This program finds the rotation angles of each edge of the convex polygon,
# then tests the area of a bounding box aligned with the unique angles in
# 90 degrees of the 1st Quadrant.
# Returns the
#
# Tested with Python 2.6.5 on Ubuntu 10.04.4
# Results verified using Matlab

# Copyright (c) 2013, David Butterworth, University of Queensland
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Willow Garage, Inc. nor the names of its
#       contributors may be used to endorse or promote products derived from
#       this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


#!/usr/bin/python

# Compute the convex hull of a set of 2D points
# A Python implementation of the qhull algorithm
#
# Tested with Python 2.6.5 on Ubuntu 10.04.4

# Copyright (c) 2008 Dave (www.literateprograms.org)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.


link = lambda a,b: np.concatenate((a,b[1:]))
edge = lambda a,b: np.concatenate(([a],[b]))

class BoundRect():
    
    def qhull2D(self, sample):
        def dome(sample,base):
            h, t = base
            dists = np.dot(sample-h, np.dot(((0,-1),(1,0)),(t-h)))
            outer = np.repeat(sample, dists>0, 0)
            if len(outer):
                pivot = sample[np.argmax(dists)]
                return link(dome(outer, edge(h, pivot)),
                        dome(outer, edge(pivot, t)))
            else:
                return base
        if len(sample) > 2:
            axis = sample[:,0]
            base = np.take(sample, [np.argmin(axis), np.argmax(axis)], 0)
            return link(dome(sample, base), dome(sample, base[::-1]))
        else:
            return sample

    def minBoundingRect(self, hull_points_2d):
        #print "Input convex hull points: "
        #print hull_points_2d
    
        # Compute edges (x2-x1,y2-y1)
        edges = np.zeros( (len(hull_points_2d)-1,2) ) # empty 2 column array
        for i in range( len(edges) ):
            edge_x = hull_points_2d[i+1,0] - hull_points_2d[i,0]
            edge_y = hull_points_2d[i+1,1] - hull_points_2d[i,1]
            edges[i] = [edge_x,edge_y]
        #print "Edges: \n", edges
    
        # Calculate edge angles   atan2(y/x)
        edge_angles = np.zeros( (len(edges)) ) # empty 1 column array
        for i in range( len(edge_angles) ):
            edge_angles[i] = math.atan2( edges[i,1], edges[i,0] )
        #print "Edge angles: \n", edge_angles
    
        # Check for angles in 1st quadrant
        for i in range( len(edge_angles) ):
            edge_angles[i] = abs( edge_angles[i] % (math.pi/2) ) # want strictly positive answers
        #print "Edge angles in 1st Quadrant: \n", edge_angles
    
        # Remove duplicate angles
        edge_angles = np.unique(edge_angles)
        #print "Unique edge angles: \n", edge_angles
    
        # Test each angle to find bounding box with smallest area
        min_bbox = (0, sys.maxsize, 0, 0, 0, 0, 0, 0) # rot_angle, area, width, height, min_x, max_x, min_y, max_y
        # print ("Testing", len(edge_angles), "possible rotations for bounding box... \n")
        for i in range( len(edge_angles) ):
        
            # Create rotation matrix to shift points to baseline
            # R = [ cos(theta)      , cos(theta-PI/2)
            #       cos(theta+PI/2) , cos(theta)     ]
            R = np.array([ [ math.cos(edge_angles[i]), math.cos(edge_angles[i]-(math.pi/2)) ],
                     [ math.cos(edge_angles[i]+(math.pi/2)), math.cos(edge_angles[i]) ] ])
            #print "Rotation matrix for ", edge_angles[i], " is \n", R
        
            # Apply this rotation to convex hull points
            rot_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
            #print "Rotated hull points are \n", rot_points
        
            # Find min/max x,y points
            min_x = np.nanmin(rot_points[0], axis=0)
            max_x = np.nanmax(rot_points[0], axis=0)
            min_y = np.nanmin(rot_points[1], axis=0)
            max_y = np.nanmax(rot_points[1], axis=0)
            #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y
        
            # Calculate height/width/area of this bounding rectangle
            width = max_x - min_x
            height = max_y - min_y
            area = width*height
            #print "Potential bounding box ", i, ":  width: ", width, " height: ", height, "  area: ", area
        
            # Store the smallest rect found first (a simple convex hull might have 2 answers with same area)
            if (area < min_bbox[1]):
                min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
            # Bypass, return the last found rect
            #min_bbox = ( edge_angles[i], area, width, height, min_x, max_x, min_y, max_y )
    
        # Re-create rotation matrix for smallest rect
        angle = min_bbox[0]
        R = np.array([ [ math.cos(angle), math.cos(angle-(math.pi/2)) ], [ math.cos(angle+(math.pi/2)), math.cos(angle) ] ])
        #print "Projection matrix: \n", R
    
        # Project convex hull points onto rotated frame
        proj_points = np.dot(R, np.transpose(hull_points_2d) ) # 2x2 * 2xn
        #print "Project hull points are \n", proj_points
    
        # min/max x,y points are against baseline
        min_x = min_bbox[4]
        max_x = min_bbox[5]
        min_y = min_bbox[6]
        max_y = min_bbox[7]
        #print "Min x:", min_x, " Max x: ", max_x, "   Min y:", min_y, " Max y: ", max_y
    
        # Calculate center point and project onto rotated frame
        center_x = (min_x + max_x)/2
        center_y = (min_y + max_y)/2
        center_point = np.dot( [ center_x, center_y ], R )
        #print "Bounding box center point: \n", center_point
    
        # Calculate corner points and project onto rotated frame
        corner_points = np.zeros( (4,2) ) # empty 2 column array
        corner_points[0] = np.dot( [ max_x, min_y ], R )
        corner_points[1] = np.dot( [ min_x, min_y ], R )
        corner_points[2] = np.dot( [ min_x, max_y ], R )
        corner_points[3] = np.dot( [ max_x, max_y ], R )
        #print "Bounding box corner points: \n", corner_points
    
        #print "Angle of rotation: ", angle, "rad  ", angle * (180/math.pi), "deg"
    
        return (angle, min_bbox[1], min_bbox[2], min_bbox[3], center_point, corner_points) # rot_angle, area, width, height, center_point, corner_points

    def getRectangle(self, xy_points):
            try:
                hull = scipy.spatial.ConvexHull(xy_points)
                hull_points = hull.vertices  
                # print(hull_points)
                hull_points = xy_points[hull_points][::-1]
                (rot_angle, area, width, height, center_point, corner_points) = self.minBoundingRect(hull_points)
                return(corner_points.T)
            except:
                return None
        
if __name__ == "__main__":
    #
    # Un-comment one of these shapes below:
    #
    
    # Square
    #xy_points = 10*array([(x,y) for x in arange(10) for y in arange(10)])
    
    # Random points
    xy_points = 100*np.random.random((32,2))
    
    # A rectangle
    #xy_points = array([ [0,0], [1,0], [1,2], [0,2], [0,0] ])
    
    # A rectangle, with 5th outlier
   # xy_points = np.array([ [0,0], [1,0], [1.5,1], [1,2], [0,2], [0,0] ])
    
    #--------------------------------------------------------------------------#
    BR = BoundRect()
    cp = BR.getRectangle(xy_points)
    print(cp)
    exit(0)
    
    # Find convex hull
    hull = scipy.spatial.ConvexHull(xy_points)
    hull_points = hull.vertices
    print(hull.vertices)
    
    # Reverse order of points, to match output from other qhull implementations
    hull_points = xy_points[hull_points][::-1]
    
    print ('Convex hull points: \n', hull_points, "\n")
    
    # Find minimum area bounding rectangle
    (rot_angle, area, width, height, center_point, corner_points) = BR.minBoundingRect(hull_points)
    
    print ("Minimum area bounding box:")
    print ("Rotation angle:", rot_angle, "rad  (", rot_angle*(180/math.pi), "deg )")
    print ("Width:", width, " Height:", height, "  Area:", area)
    print ("Center point: \n", center_point )# numpy array
    print ("Corner points: \n", corner_points, "\n" ) # numpy arrayarray([ [0,0], [1,0], [1.5,1], [1,2], [0,2], [0,0] ])
    print(xy_points)
    mpl.plot(xy_points.T[0], xy_points.T[1], 'ro')
    mpl.plot(xy_points[hull.vertices, 0], xy_points[hull.vertices, 1], 'c-')
    mpl.plot(corner_points.T[0], corner_points.T[1], 'k-')
    mpl.plot(hull_points.T, 'r-')
    mpl.show()