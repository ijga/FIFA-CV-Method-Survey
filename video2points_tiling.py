from ultralytics import YOLO
import numpy as np
import cv2
from skimage.io import imread, imshow
import matplotlib.pyplot as plt
from graph import Graph, Player, Edge, Ball, Goal
from classes import class_names
from tiling import predict_tiling, find_bottom
from tiling_multiprocessing import parallel_predict_tiling
import time


def run():
   
    model = YOLO('models/best7_coop.pt')

    np.set_printoptions(threshold = np.inf)

    # 1080
    Hs = np.array([[ 5.94007434e-01,  1.18369697e+00,  3.77156029e+02],
                [-1.29464862e-15,  2.22727388e+00, -1.82615885e+02],
                [-7.25486452e-19,  1.24457088e-03,  1.00000000e+00]])
    
    # results = model("videos/coop/game1.mov", imgsz=1920, show=True)

    cap = cv2.VideoCapture("videos/coop/game1.mov")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    results = parallel_predict_tiling(cap, 10)

    for idx, result in enumerate(results):
        
        graph = Graph()
        points = []

        bboxes, confs, class_ids, fps, frame = result.values()

        # xyxy
        top_lefts = {}
        top_rights = {}
        bottom_lefts = {}
        bottom_rights = {}
        areas = {}

        skipped = set()

        for i in range(len(bboxes)):

            if class_ids[i] == 0:
                continue

            area = (bboxes[i][2] - bboxes[i][0]) *  (bboxes[i][3] - bboxes[i][1])

            top_left = (bboxes[i][0]//25, bboxes[i][1]//25)
            top_right = (bboxes[i][0]//25, bboxes[i][3]//25)
            bottom_left = (bboxes[i][2]//25, bboxes[i][1]//25)
            bottom_right = (bboxes[i][2]//25, bboxes[i][3]//25)

            if top_left in top_lefts:
                if areas[top_lefts[top_left]] >= area:
                    skipped.add(i)
                else:  
                    skipped.add(top_lefts[top_left])
                    top_lefts[top_left] = i
                    top_rights[top_right] = i
                    bottom_lefts[bottom_left] = i
                    bottom_rights[bottom_right] = i
                    areas[i] = area
                continue
            if top_right in top_rights:
                if areas[top_rights[top_right]] >= area:
                    skipped.add(i)
                else:  
                    skipped.add(top_rights[top_right])
                    top_lefts[top_left] = i
                    top_rights[top_right] = i
                    bottom_lefts[bottom_left] = i
                    bottom_rights[bottom_right] = i
                    areas[i] = area
                continue
            if bottom_left in bottom_lefts:
                if areas[bottom_lefts[bottom_left]] >= area:
                    skipped.add(i)
                else:  
                    skipped.add(bottom_lefts[bottom_left])
                    top_lefts[top_left] = i
                    top_rights[top_right] = i
                    bottom_lefts[bottom_left] = i
                    bottom_rights[bottom_right] = i
                    areas[i] = area
                continue
            if bottom_right in bottom_rights:
                if areas[bottom_rights[bottom_right]] >= area:
                    skipped.add(i)
                else:  
                    skipped.add(bottom_rights[bottom_right])
                    top_lefts[top_left] = i
                    top_rights[top_right] = i
                    bottom_lefts[bottom_left] = i
                    bottom_rights[bottom_right] = i
                    areas[i] = area
                continue

            top_lefts[top_left] = i
            top_rights[top_right] = i
            bottom_lefts[bottom_left] = i
            bottom_rights[bottom_right] = i
            areas[i] = area

        cleaned_class_ids = []
        for i in range(len(bboxes)):
            if i in skipped:
                continue
            x1, y1, x2, y2 = bboxes[i]

            object_point = find_bottom(bboxes[i])
            points.append(object_point)

            conf = confs[i]
            class_id = class_ids[i]
            cleaned_class_ids.append(class_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.1f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f'{number_of_tiles}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'{fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        points = np.array(points).reshape(1, -1, 2)
        
        transformed_points = np.array([])
        try:
            transformed_points = cv2.perspectiveTransform(points, Hs)
            if len(transformed_points) > 0: # error thrown if null
                pass
        except:
            transformed_points = np.array([])

        img = np.ones((1080, 1920, 3), dtype=np.uint8)  # 1080p
        img = 200 * img # make it gray

        # clustering on points
        if len(transformed_points) > 0:
            idx = 0
            for type, point in zip(cleaned_class_ids, transformed_points[0, :, :]):
                type = int(type)
                
                if class_names[type] == 'man_city' or class_names[type] == 'man_utd':
                    graph.add_player(Player(idx, class_names[type], point[0], point[1]))
                elif class_names[type] == 'man_city_gk' or class_names[type] == 'man_utd_gk':
                    graph.add_gk(Player(idx, class_names[type], point[0], point[1]))
                elif class_names[type] == 'ball':
                    graph.add_ball(Ball(idx, point[0], point[1]))
                elif class_names[type] == 'left_goal' or class_names[type] == 'right_goal':
                    graph.add_goal(Goal(idx, type, point[0], point[1]))

                if type == 4: # man_city
                    cv2.circle(img, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)  # Draw a blue circle at each point
                elif type == 6: # man_utd
                    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)  # Draw a red circle at each point
                elif type == 0: # ball orange
                    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 140, 255), -1)  # Draw a orange circle at each point
                else: # anything else
                    cv2.circle(img, (int(point[0]), int(point[1])), 3, (0, 255, 0), -1)  # Draw a green circle at each point
                
                idx += 1
            graph.add_edges(4)
            graph.visualize_graph()
            cv2.imshow('game items', img)

        # Write the frame into the file 'output.mp4'
        out.write(frame)

        cv2.imshow('frame', frame)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()