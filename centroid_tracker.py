from collections import OrderedDict
import numpy as np
import math


class CentroidTrackerWithThreshold:
    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 0


    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object
        for rect in objects_rect:
            x, y, x2, y2 = rect
            w, h = x2 - x, y2 - y
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 70:
                    self.center_points[id] = (cx, cy)
                    #print(self.center_points)
                    objects_bbs_ids.append([x, y, x2, y2, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, x2, y2, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()
        #return objects_bbs_ids
        obj_ids = {
            i[4]: np.array([int((i[0] + i[2]) / 2), int((i[1] + i[3]) / 2)])
            for i in objects_bbs_ids
        }
        obj_rects = {
            i[4]: np.array([i[0], i[1], i[2], i[3]])
            for i in objects_bbs_ids
        }
        return obj_ids, obj_rects


class CentroidTrackerWithHistory():
    def __init__(self, maxDisappeared=5):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.originRects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
    

    def get_iou(self, bb1, bb2):
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def euclidian(self, xA, xB):
        return np.sqrt(
            (xA[0] - xB[0])**2 + (xA[1] - xB[1])**2
        )

    def register(self, centroid, rect):
        self.originRects[self.nextObjectID] = rect
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.originRects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def get_id(self, rect):
        (x, y, eX, eY) = rect
        cX = int((x + eX) / 2)
        cY = int((y + eY) / 2)

        objectIDs = list(self.objects.keys())
        objectCentroids = list(self.objects.values())
        objectRects = list(self.originRects.values())
        
        D_iou = np.array([
            self.get_iou({'x1': x, 'x2': eX, 'y1': y, 'y2': eY}, {'x1': rect[0], 'x2': rect[2], 'y1': rect[1], 'y2': rect[3]})
            for rect in objectRects
        ])

        D = np.array([self.euclidian(obj, (cX, cY)) for obj in objectCentroids])
        #dist.cdist(np.array(objectCentroids), [(cX, cY)])
        #D = D * (1-D_iou)
        #D = 1 - D_iou

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]
        objectID = None

        for (row, col) in zip(rows, cols):
            objectID = objectIDs[row]
            break
        return objectID

    def update(self, rects):

        if(len(rects) == 0):
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                
                if(self.disappeared[objectID] > self.maxDisappeared):
                    self.deregister(objectID)
            return self.objects, self.originRects
        
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for(i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2)
            cY = int((startY + endY) / 2)
            inputCentroids[i] = (cX, cY)

        if(len(self.objects) == 0):
            for i in range(0, len(inputCentroids)):
                centroid = inputCentroids[i]
                rect = rects[i]
                self.register(centroid, rect)
        
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            objectRects = list(self.originRects.values())

            D_iou = np.array([
                [
                    self.get_iou(
                        {'x1': rect[0], 'x2': rect[2], 'y1': rect[1], 'y2': rect[3]}, 
                        {'x1': objectRect[0], 'x2': objectRect[2], 'y1': objectRect[1], 'y2': objectRect[3]}
                    ) for rect in rects]
                for objectRect in objectRects
            ])

            D = np.array([
                [
                    self.euclidian(i, j) for i in inputCentroids]
                for j in objectCentroids
            ])

            #D = D * (1 - D_iou)
            #D = 1 - D_iou

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.originRects[objectID] = rects[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:

                for col in unusedCols:
                    centroid = inputCentroids[col]
                    rect = rects[col]
                    self.register(centroid, rect)

        return self.objects, self.originRects
