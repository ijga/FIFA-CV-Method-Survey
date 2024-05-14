import cv2
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
from ultralytics import YOLO
import time
import torch.multiprocessing as mp
import torch
import queue
import os

class SlicedImage:
    def __init__(self, image, starting_pixel):
        self.image = image
        self.starting_pixel = starting_pixel


class SliceImageResult:
    def __init__(self, original_image_size: List[int], image_dir: Optional[str] = None):
        self.original_image_height = original_image_size[0]
        self.original_image_width = original_image_size[1]
        self.image_dir = image_dir

        self._sliced_image_list: List[SlicedImage] = []

    def add_sliced_image(self, sliced_image: SlicedImage):
        if not isinstance(sliced_image, SlicedImage):
            raise TypeError("sliced_image must be a SlicedImage instance")

        self._sliced_image_list.append(sliced_image)

    @property
    def sliced_image_list(self):
        return self._sliced_image_list

    @property
    def images(self):
        images = []
        for sliced_image in self._sliced_image_list:
            images.append(sliced_image.image)
        return images

    @property
    def starting_pixels(self) -> List[int]:
        starting_pixels = []
        for sliced_image in self._sliced_image_list:
            starting_pixels.append(sliced_image.starting_pixel)
        return starting_pixels

    @property
    def filenames(self) -> List[int]:
        filenames = []
        for sliced_image in self._sliced_image_list:
            filenames.append(sliced_image.coco_image.file_name)
        return filenames

    def __getitem__(self, i):
        def _prepare_ith_dict(i):
            return {
                "image": self.images[i],
                "starting_pixel": self.starting_pixels[i],
            }

        if isinstance(i, np.ndarray):
            i = i.tolist()

        if isinstance(i, int):
            return _prepare_ith_dict(i)
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return [_prepare_ith_dict(i) for i in range(start, stop, step)]
        elif isinstance(i, (tuple, list)):
            accessed_mapping = map(_prepare_ith_dict, i)
            return list(accessed_mapping)
        else:
            raise NotImplementedError(f"{type(i)}")

    def __len__(self):
        return len(self._sliced_image_list)


def slice_image(
    image: Union[str, Image.Image],
) -> SliceImageResult:

    image_pil = image
    slice_bboxes = [[0, 0, 540, 640], [486, 0, 1026, 640], [972, 0, 1512, 640], [1380, 0, 1920, 640], [0, 440, 540, 1080], [486, 440, 1026, 1080], [972, 440, 1512, 1080], [1380, 440, 1920, 1080]]

    n_ims = 0

    sliced_image_result = SliceImageResult(original_image_size=[1080, 1920])

    image_pil_arr = np.asarray(image_pil)

    for slice_bbox in slice_bboxes:
        n_ims += 1

        tlx, tly, brx, bry = slice_bbox
        image_pil_slice = image_pil_arr[tly:bry, tlx:brx]
   
        sliced_image = SlicedImage(
            image=image_pil_slice, starting_pixel=[slice_bbox[0], slice_bbox[1]]
        )
        sliced_image_result.add_sliced_image(sliced_image)

    return sliced_image_result


def process_tile(in_queue, out_queue, slice_box):
    model = YOLO('models/best7_coop.pt')

    while True:
        try:
            sliced_image = in_queue.get()

            start_frame_time = time.perf_counter()
            process_id = os.getpid()
            results = []
            bboxes = []
            confs = []
            class_ids = []

            window = sliced_image['image']
            # start_x, start_y = sliced_image['starting_pixel']
            start_x, start_y = sliced_image['starting_pixel']

            results = model.predict(window, conf=0.7)

            for result in results:

                boxes = result.boxes  # Boxes object for bounding box outputs

                xyxy = boxes.xyxy.numpy()

                if xyxy.size == 0:
                    continue
                
                xyxy = xyxy
                conf = boxes.conf.numpy()
                class_id = boxes.cls.numpy()

                for i in range(len(xyxy)):

                    x1, y1, x2, y2 = xyxy[i]

                    x1 += start_x
                    y1 += start_y
                    x2 += start_x
                    y2 += start_y
                
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    bboxes.append([x1, y1, x2, y2])
                    confs.append(conf[i])
                    class_ids.append(class_id[i])

            end_frame_time = time.perf_counter()
            execution_time = end_frame_time - start_frame_time
            # print("process_tile Time:", execution_time)
            # print(process_id)
        
        except queue.Empty:
            continue
            
        else:
            print("task" + ' is done by ' + mp.current_process().name)
            out_queue.put({"bboxes": bboxes, "confs": confs, "class_id": class_id, "core": process_id, "time": execution_time})
    
    return True

def parallel_predict_tiling(cap, frequency=40, height_fraction=3, width_fraction=2):
    
    # torch.set_num_threads(1)

    # Get the width and height of the video frame
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0

    num_processes = 8
    in_queue = mp.Queue()
    out_queue = mp.Queue(maxsize=8)

    processes = []
    slice_bboxes = [[0, 0, 540, 640], [486, 0, 1026, 640], [972, 0, 1512, 640], [1380, 0, 1920, 640], [0, 440, 540, 1080], [486, 440, 1026, 1080], [972, 440, 1512, 1080], [1380, 440, 1920, 1080]]
    for w in range(num_processes):
        p = mp.Process(target=process_tile, args=(in_queue, out_queue, slice_bboxes[w]))
        p.Daemon = True
        processes.append(p)
        p.start()

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

        # image_pil_arr = np.asarray(image) # share this across all threads

        slice_image_result = slice_image(
            image=image,
        )
        
        print("start pooling")
        start_pool= time.perf_counter()

        for tile in slice_image_result:
            in_queue.put(tile)

        while not out_queue.full():
            pass

        bboxes = []
        confs = []
        class_ids = []

        print("\nQUEUE FULL\n")

        while not out_queue.empty():
            intermediate = out_queue.get()
            bboxes.extend(intermediate["bboxes"])
            confs.extend(intermediate["confs"])
            class_ids.extend(intermediate["class_id"])

        # print({
        #     "bboxes": bboxes,
        #     "confs": confs,
        #     "class_ids": class_ids,
        # })

        print("end pooling")

        end_pool = time.perf_counter()
        print("pooling Time:", end_pool - start_pool)

        end_time = time.perf_counter()
        total_time = end_time - start_time
        fps = 1 / total_time

        end_frame_time = time.perf_counter()
        execution_time = end_frame_time - start_frame_time
        print("Execution Time:", execution_time)

        yield {
            "bboxes": bboxes,
            "confs": confs,
            "class_ids": class_ids,
            "fps": fps,
            "frame": frame
        }

"""finds middle of the bottom of the bounding box, used as the point where the object is located (besides goals)"""
def find_bottom(xyxy):
    return [(xyxy[0]+ xyxy[2])/2, xyxy[3]]