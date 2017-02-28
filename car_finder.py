import math
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog

from lane_finder import LaneFinder
from settings import CALIB_FILE_NAME, PERSPECTIVE_FILE_NAME, UNWARPED_SIZE, ORIGINAL_SIZE


class DigitalFilter:

    def __init__(self, vector, b, a):
        self.len = len(vector)
        self.b = b.reshape(-1, 1)
        self.a = a.reshape(-1, 1)
        self.input_history = np.tile(vector.astype(np.float64), (len(self.b), 1))
        self.output_history = np.tile(vector.astype(np.float64), (len(self.a), 1))
        self.old_output = np.copy(self.output_history[0])

    def output(self):
        return self.output_history[0]

    def speed(self):
        return self.output_history[0] - self.output_history[1]

    def new_point(self, vector):
        self.input_history = np.roll(self.input_history, 1, axis=0)
        self.old_output = np.copy(self.output_history[0])
        self.output_history = np.roll(self.output_history, 1, axis=0)
        self.input_history[0] = vector
        self.output_history[0] = (np.matmul(self.b.T, self.input_history) - np.matmul(self.a[1:].T, self.output_history[1:]))/self.a[0]
        return self.output()

    def skip_one(self):
        self.new_point(self.output())


def area(bbox):
    return float((bbox[3] - bbox[1]) * (bbox[2] - bbox[0]))


class Car:
    def __init__(self, bounding_box, first=False, warped_size=None, transform_matrix=None, pix_per_meter=None):
        self.warped_size = warped_size
        self.transform_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter
        self.has_position = self.warped_size is not None \
                            and self.transform_matrix is not None \
                            and self.pix_per_meter is not None

        self.filtered_bbox = DigitalFilter(bounding_box, 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.position = DigitalFilter(self.calculate_position(bounding_box), 1/21*np.ones(21, dtype=np.float32), np.array([1.0, 0]))
        self.found = True
        self.num_lost = 0
        self.num_found = 0
        self.display = first
        self.fps = 25

    def calculate_position(self, bbox):
        if (self.has_position):
            pos = np.array((bbox[0]/2+bbox[2]/2, bbox[3])).reshape(1, 1, -1)
            dst = cv2.perspectiveTransform(pos, self.transform_matrix).reshape(-1, 1)
            return np.array((self.warped_size[1]-dst[1])/self.pix_per_meter[1])
        else:
            return np.array([0])

    def get_window(self):
        return self.filtered_bbox.output()

    def one_found(self):
        self.num_lost = 0
        if not self.display:
            self.num_found += 1
            if self.num_found > 5:
                self.display = True

    def one_lost(self):
        self.num_found = 0
        self.num_lost += 1
        if self.num_lost > 5:
            self.found = False

    def update_car(self, bboxes):
        current_window = self.filtered_bbox.output()
        intersection = np.zeros(4, dtype = np.float32)
        for idx, bbox in enumerate(bboxes):
            intersection[0:2] = np.maximum(current_window[0:2], bbox[0:2])
            intersection[2:4] = np.minimum(current_window[2:4], bbox[2:4])
            if (area(bbox)>0) and area(current_window) and ((area(intersection)/area(current_window)>0.8) or (area(intersection)/area(bbox)>0.8)):
                self.one_found()
                self.filtered_bbox.new_point(bbox)
                self.position.new_point(self.calculate_position(bbox))
                bboxes.pop(idx)
                return

        self.one_lost()
        self.filtered_bbox.skip_one()
        self.position.skip_one()

    def draw(self, img, color=(255, 0, 0), thickness=2):
        if self.display:
            window = self.filtered_bbox.output().astype(np.int32)
            cv2.rectangle(img, (window[0], window[1]), (window[2], window[3]), color, thickness)
            if self.has_position:
                cv2.putText(img, "RPos: {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "RPos: {:6.2f}m".format(self.position.output()[0]), (int(window[0]), int(window[1]-5)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=3, color=(255, 255, 255))
                cv2.putText(img, "RVel: {:6.2f}km/h".format(self.position.speed()[0]*self.fps*3.6), (int(window[0]), int(window[3]+20)),
                            cv2.FONT_HERSHEY_PLAIN, fontScale=1.25, thickness=2, color=(0, 0, 0))


class CarFinder:
    def __init__(self, size, hist_bins, small_size, orientations=12, pix_per_cell=8, cell_per_block=2,
                 hist_range=None, scaler=None, classifier=None, window_sizes=None, window_rois=None,
                 warped_size=None, transform_matrix=None, pix_per_meter=None):
        self.size = size
        self.small_size = small_size
        self.hist_bins = hist_bins
        self.hist_range = (0, 256)
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.orientations = orientations
        self.scaler = scaler
        self.cls = classifier
        self.num_cells = self.size//self.pix_per_cell
        self.num_blocks = self.num_cells - (self.cell_per_block-1)
        self.num_features = self.calc_num_features()
        if hist_range is not None:
            self.hist_range = hist_range

        self.window_sizes = window_sizes
        self.window_rois = window_rois
        self.cars = []
        self.first = True
        self.warped_size = warped_size
        self.transformation_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter

    def calc_num_features(self):
        return self.small_size**2*3 + self.hist_bins*3 + 3*self.num_blocks**2 *self.cell_per_block**2*self.orientations

    def get_features(self, img):
        img_resize = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img_resize = (np.sqrt(img_resize.astype(np.float32)/255)*255).astype(np.uint8)
        img_LUV = cv2.cvtColor(img_resize, cv2.COLOR_RGB2LUV)
        img_feature = cv2.resize(img_LUV, (self.small_size, self.small_size), interpolation=cv2.INTER_LINEAR)
        hist_l = np.histogram(img_LUV[:, :, 0], bins=self.hist_bins, range=self.hist_range)
        width = 0.7 * (hist_l[1][1] - hist_l[1][0])
        center = (hist_l[1][:-1] + hist_l[1][1:]) / 2
        hist_u = np.histogram(img_LUV[:, :, 1], bins=self.hist_bins, range=self.hist_range)
        hist_v = np.histogram(img_LUV[:, :, 2], bins=self.hist_bins, range=self.hist_range)
        features_l = hog(img_LUV[:, :, 0], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                       feature_vector=True)
        features_u = hog(img_LUV[:, :, 1], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                       feature_vector=True)
        features_v = hog(img_LUV[:, :, 2], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                       feature_vector=True)
        return np.hstack((img_feature.ravel(), hist_l[0], hist_u[0], hist_v[0], features_l, features_u, features_v))

    def car_find_roi(self, img, size, roi, overlap):
        assert self.scaler is not None, "CarFinder error -> Scaler has to be initialized"
        assert self.cls is not None, "CarFinder error -> Classifier has to be initialized"
        width = roi[1][0]-roi[0][0]
        height = roi[1][1]-roi[0][1]
        img_roi = img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]
        scale = float(size)/self.size
        new_width = int(math.ceil(float(width)/scale))
        new_height = int(math.ceil(float(height)/scale))
        img_roi = cv2.resize(img_roi, (new_width, new_height))
        img_roi = (np.sqrt(img_roi.astype(np.float32)/255)*255).astype(np.uint8)
        img_roi = cv2.cvtColor(img_roi, cv2.COLOR_RGB2LUV)
        img_small = cv2.resize(img_roi, (math.ceil(new_width*self.small_size/float(self.size)),
                                         math.ceil(new_height*self.small_size/float(self.size))))
        img_hog_l = hog(img_roi[:, :, 0], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                      cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                      feature_vector=False)
        img_hog_u = hog(img_roi[:, :, 1], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                      cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                      feature_vector=False)
        img_hog_v = hog(img_roi[:, :, 2], self.orientations, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                      cells_per_block=(self.cell_per_block,  self.cell_per_block), transform_sqrt=False,
                      feature_vector=False)
        shift_roi = int((1-overlap)*self.size)
        shift_small = int((1-overlap)*self.small_size)
        shift_hog = int((1-overlap)*(self.size//self.pix_per_cell))
        n_horizontal = int((new_width-self.size)/shift_roi) + 1
        n_vertical = int((new_height-self.size)/shift_roi) + 1
        total_windows = n_horizontal*n_vertical

        all_features = np.zeros((total_windows, self.num_features), dtype = np.float32)
        all_coordinates = np.zeros((total_windows, 4), dtype=np.int32)

        current = 0
        for col in range(n_horizontal):
            for row in range(n_vertical):
                img_ = img_roi[row*shift_roi:row*shift_roi+self.size, col*shift_roi:col*shift_roi+self.size]
                hist_h = np.histogram(img_[:, :, 0], bins=self.hist_bins, range=self.hist_range)
                hist_l = np.histogram(img_[:, :, 1], bins=self.hist_bins, range=self.hist_range)
                hist_s = np.histogram(img_[:, :, 2], bins=self.hist_bins, range=self.hist_range)
                all_features[current] = np.hstack((img_small[row*shift_small:row*shift_small+self.small_size,
                                                   col*shift_small:col*shift_small+self.small_size].ravel(),
                                                   hist_h[0], hist_l[0], hist_s[0],
                                                   img_hog_l[row*shift_hog:row*shift_hog+self.num_cells,
                                                   col*shift_hog:col*shift_hog+self.num_cells].ravel(),
                                                   img_hog_u[row*shift_hog:row*shift_hog+self.num_cells,
                                                   col*shift_hog:col*shift_hog+self.num_cells].ravel(),
                                                   img_hog_v[row*shift_hog:row*shift_hog+self.num_cells,
                                                   col*shift_hog:col*shift_hog+self.num_cells].ravel()
                                                   ))
                all_coordinates[current][0] = roi[0][0] + int(scale*col*shift_roi)
                all_coordinates[current][1] = roi[0][1] + int(scale*row*shift_roi)
                current += 1

        all_coordinates[:, 2:4] = size + all_coordinates[:, 0:2]
        cars = self.cls.predict(self.scaler.transform(all_features))
        return all_coordinates[cars == 1].tolist()


    def find_cars(self, img, threshold = 1, reset = False):
        heatmap = np.zeros_like(img[:,:,1])
        car_windows = []
        if reset:
            self.cars = []
            self.first = True
        for size, roi in zip(self.window_sizes, self.window_rois):
            car_windows += self.car_find_roi(img, size, roi, overlap=0.75)
        for car in self.cars:
            window = car.get_window()
            heatmap[window[1]:window[3],window[0]:window[2]] += 1

        for window in car_windows:
            heatmap[window[1]:window[3],window[0]:window[2]] += 1

        heatmap = heatmap > threshold
        label_img, labels = label(heatmap)
        bboxes = []
        for lbl in range(labels):
            points = (label_img == lbl+1).nonzero()
            nonzeroy = np.array(points[0])
            nonzerox = np.array(points[1])
            bbox = np.array((np.min(nonzerox), np.min(nonzeroy), np.max(nonzerox), np.max(nonzeroy)))
            car_img = img[bbox[1]:bbox[3],bbox[0]:bbox[2], :]
            he = bbox[3]-bbox[1]
            medi = np.median(car_img[-he//8:-1], axis=[0,1])
            near = cv2.inRange(car_img, medi - np.array([35, 35, 35]),medi+np.array([35, 35, 35]))
            if near is not None:
                cc = np.sum(near, axis=1)/255 > int(0.8*near.shape[1])
                eee = len(cc)-1
                while eee >= 0 and cc[eee]:
                   eee -= 1
                bbox[3] = bbox[1]+eee
            bboxes.append(bbox)

        for car in self.cars:
            car.update_car(bboxes)

        for bbox in bboxes:
            self.cars.append(Car(bbox, self.first, self.warped_size, self.transformation_matrix, self.pix_per_meter))

        tmp_cars = []
        for car in self.cars:
            if car.found:
                tmp_cars.append(car)
        self.cars = tmp_cars
        self.first = False

    def draw_cars(self, img):
        i2 = np.copy(img)
        for car in self.cars:
            car.draw(i2)
        return i2

if __name__ == '__main__':
    with open('classifier.p', 'rb') as f:
        data = pickle.load(f)

    scaler = data['scaler']
    cls = data['classifier']

    test_images_dir = "test_images"
    output_images_dir = "output_images"
    window_size=[64, 80, 96, 112, 128, 160]#, 192, 224, 256]
    window_roi=[((200, 400),(1080, 550)), ((100, 400),(1180, 550)), ((0, 380),(1280, 550)),
                ((0, 360),(1280, 550)), ((0, 360),(1280, 600)), ((0, 360),(1280, 670)) ]#,

    with open(CALIB_FILE_NAME, 'rb') as f:
        calib_data = pickle.load(f)
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]
        img_size = calib_data["img_size"]

    with open(PERSPECTIVE_FILE_NAME, 'rb') as f:
        perspective_data = pickle.load(f)

    perspective_transform = perspective_data["perspective_transform"]
    pixels_per_meter = perspective_data['pixels_per_meter']
    orig_points = perspective_data["orig_points"]


    video_files = ['test_video.mp4', 'project_video.mp4']
    output_path = "output_videos"

    def process_image(img, car_finder, lane_finder, cam_matrix, dist_coeffs, reset = False):
        img = cv2.undistort(img, cam_matrix, dist_coeffs)
        car_finder.find_cars(img, reset = reset)
        lane_finder.find_lane(img, distorted=False, reset=reset)
        return lane_finder.draw_lane_weighted(car_finder.draw_cars(img))

    lf = LaneFinder(ORIGINAL_SIZE, UNWARPED_SIZE, cam_matrix, dist_coeffs,
                        perspective_transform, pixels_per_meter, "warning.png")
    cf = CarFinder(64, hist_bins=128, small_size=20, orientations=12, pix_per_cell=8, cell_per_block=1,
                       classifier=cls, scaler=scaler, window_sizes=window_size, window_rois=window_roi,
                       transform_matrix=perspective_transform, warped_size=UNWARPED_SIZE,
                       pix_per_meter=pixels_per_meter)

    for img_path in os.listdir(test_images_dir):
        if "jpg" in img_path:
            img = mpimg.imread(os.path.join(test_images_dir, img_path))
            res_img = process_image(img, cf, lf, cam_matrix, dist_coeffs, True)
            plt.imshow(res_img)
            plt.show()
            mpimg.imsave(os.path.join(output_images_dir, img_path), res_img)

    for file in video_files:
        lf = LaneFinder(ORIGINAL_SIZE, UNWARPED_SIZE, cam_matrix, dist_coeffs,
                        perspective_transform, pixels_per_meter, "warning.png")
        cf = CarFinder(64, hist_bins=128, small_size=20, orientations=12, pix_per_cell=8, cell_per_block=1,
                           classifier=cls, scaler=scaler, window_sizes=window_size, window_rois=window_roi,
                               transform_matrix=perspective_transform, warped_size=UNWARPED_SIZE,
                               pix_per_meter=pixels_per_meter)
        output = os.path.join(output_path,"cars_"+file)
        clip2 = VideoFileClip(file)
        challenge_clip = clip2.fl_image(lambda x: process_image(x, cf, lf, cam_matrix, dist_coeffs))
        challenge_clip.write_videofile(output, audio=False)





