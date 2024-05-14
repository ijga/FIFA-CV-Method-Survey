from ultralytics import YOLO
import numpy as np
import cv2
from graph import Graph, Player, Edge, Ball, Goal
from classes import class_names
import time
from PIL import Image

def whole_vision(cap, frequency=40):
    
    model = YOLO('models/best7_coop.pt')

    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0

    while cap.isOpened():
        start_frame_time = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            break

        count += 1
        if count % frequency != 0:
            continue 

        start_time = time.perf_counter()
        
        image = Image.fromarray(frame)
        height, width = image.size

        result = model.predict(image, conf=0.7, imgsz=1920)
        # print(result[0])
        boxes = result[0].boxes

        xyxy = boxes.xyxy.numpy()
        confs = boxes.conf.numpy()
        classes = boxes.cls.numpy()

        for i in range(len(xyxy)):
            x1, y1, x2, y2 = xyxy[i]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = classes[i]
            conf = confs[i]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{model.names[int(class_id)]} {conf:.1f}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # cv2.putText(frame, f'{fps:.1f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        boxes = result[0].boxes.xywh.numpy()[:, :2] 
        boxes[:, 1] += (result[0].boxes.xywh.numpy()[:, 3] * 0.5)
        boxes = boxes.reshape(1, -1, 2)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = 1 / total_time

        end_frame_time = time.perf_counter()
        execution_time = end_frame_time - start_frame_time
        print(f'Execution Time {count}: {execution_time}')

        yield {
            "bboxes": boxes,
            "confs": confs,
            "class_ids": classes,
            "fps": fps,
            "frame": frame
        }


def run():
   
    model = YOLO('models/best7_coop.pt')
    np.set_printoptions(threshold = np.inf)

    results = model(source="videos/coop/game1.mov", show=True, imgsz=1920, stream=True, vid_stride=90)

    # 1080
    Hs = np.array([[ 5.94007434e-01,  1.18369697e+00,  3.77156029e+02],
                [-1.29464862e-15,  2.22727388e+00, -1.82615885e+02],
                [-7.25486452e-19,  1.24457088e-03,  1.00000000e+00]])
    
    cap = cv2.VideoCapture("videos/coop/game1.mov")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    # results = whole_vision(cap, 1)

    for idx, result in enumerate(results):

        # points, confs, classes, fps, frame = result.values()

        confidence = result.boxes.conf.numpy()
        classes = result.boxes.cls.numpy()

        boxes = result.boxes.xywh.numpy()[:, :2] 
        boxes[:, 1] += (result.boxes.xywh.numpy()[:, 3] * 0.5)
        points = boxes.reshape(1, -1, 2)
        
        transformed_points = np.array([])
        try:
            transformed_points = cv2.perspectiveTransform(points, Hs)
            if len(transformed_points) > 0: # error thrown if null
                pass
        except:
            transformed_points = np.array([])

        img = np.ones((1080, 1920, 3), dtype=np.uint8)  # 1080p
        img = 200 * img # make it gray

        graph = Graph()
        if len(transformed_points) > 0:
            idx = 0
            for type, point in zip(classes, transformed_points[0, :, :]):
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
            
        # Write the frame into the file 'output.mp4'
        # out.write(frame)
        if len(transformed_points) > 0:
            # graph.add_edges(4)
            # graph.visualize_graph()
            cv2.imshow('game items', img)

        # cv2.imshow('frame', frame)
        #cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    # out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()