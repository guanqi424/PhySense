import os
from queue import Queue
from threading import Thread
import concurrent.futures

import pandas as pd
import xgboost as xgb
import numpy as np
from PIL import Image
from skimage import feature
from sklearn.preprocessing import LabelEncoder


def read_image_regions(img_PIL, region_size_ratio, num_regions, edge_buffer_ratio):
    data = np.array(img_PIL)

    rows, cols = data.shape
    regions = []
    
    region_width = int(cols * region_size_ratio[0])
    region_height = int(rows * region_size_ratio[1])
    
    edge_buffer_width = int(cols * edge_buffer_ratio[0])
    edge_buffer_height = int(rows * edge_buffer_ratio[1])
    
    start_rows = np.random.randint(edge_buffer_height, rows - region_height - edge_buffer_height, size=num_regions)
    start_cols = np.random.randint(edge_buffer_width, cols - region_width - edge_buffer_width, size=num_regions)
    
    regions = [data[row:row + region_height, col:col + region_width] for row, col in zip(start_rows, start_cols)]
    return regions

def apply_lbp(img):
    radius = 2
    n_point = 16 
    lbp = feature.local_binary_pattern(img, n_point, radius, 'default')
    hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
    return hist

class phySenseTexture:
    def __init__(self, model_path, label_encoder_path, num_regions, region_size_ratio, edge_buffer_ratio, workers_fraction):
        self.PIPELINE_WORKERS_CNT = int(os.cpu_count() / workers_fraction)
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.load(label_encoder_path, allow_pickle=True)

        self.num_regions = num_regions
        self.region_size_ratio = region_size_ratio
        self.edge_buffer_ratio = edge_buffer_ratio

        # Separate queues for each stage
        self.image_queue = Queue()
        self.feature_queue = Queue()
        self.prediction_queue = Queue()
        self.final_result_queue = Queue()
        
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=None)

        # Start threads once and keep them alive
        self.reader_threads = []
        self.extractor_threads = []
        self.predictor_threads = []
        for i in range(self.PIPELINE_WORKERS_CNT):
            reader_thread = Thread(target=self.image_reader_worker)
            extractor_thread = Thread(target=self.feature_extractor_worker)
            predictor_thread = Thread(target=self.predictor_worker)
            
            reader_thread.start()
            extractor_thread.start()
            predictor_thread.start()

            self.reader_threads.append(reader_thread)
            self.extractor_threads.append(extractor_thread)
            self.predictor_threads.append(predictor_thread)


    def image_reader_worker(self):
        while True:
            img_lists = self.image_queue.get()
            if img_lists is None:  # Sentinel for shutdown
                break
            for img_PIL in img_lists:
                image_regions = read_image_regions(img_PIL, self.region_size_ratio, self.num_regions, self.edge_buffer_ratio)
                self.feature_queue.put(image_regions)
        self.feature_queue.put(None)  # Signal feature extractor to shutdown

    def feature_extractor_worker(self):
        while True:
            image_regions = self.feature_queue.get()
            if image_regions is None:  # Sentinel for shutdown
                break
            
            histograms = self.executor.map(apply_lbp, image_regions)
            
            hist_sum = np.sum(histograms, axis=0)
            hist_sum = hist_sum / np.sum(hist_sum)

            self.prediction_queue.put(hist_sum)
        self.prediction_queue.put(None)  # Signal predictor to shutdown

    def predictor_worker(self):
        while True:
            features = self.prediction_queue.get()
            if features is None:  # Sentinel for shutdown
                break
            probabilities = self.model.predict_proba([features])
            self.final_result_queue.put(probabilities[0])  # Put the final result in the new queue

    def predict_prob(self, img_lists):
        results = []

        # Pass images to the pipeline
        self.image_queue.put(img_lists)

        # Collect results in the same order as the images were passed
        for _ in img_lists:
            result = self.final_result_queue.get()  # Retrieve from the final result queue
            results.append(result)

        prob_df = pd.DataFrame(results, columns=self.label_encoder.classes_)
        return prob_df

    def shutdown(self):
        # Signal all threads to shutdown
        for i in range(self.PIPELINE_WORKERS_CNT):
            self.image_queue.put(None)
        for i in range(self.PIPELINE_WORKERS_CNT):
            self.reader_threads[i].join()
            # No need to signal feature_queue or prediction_queue, as they will receive None automatically
            self.extractor_threads[i].join()
            self.predictor_threads[i].join()
        self.executor.shutdown(wait=True)
    
    def __del__(self):
        print("Shutdown pipeline")
        self.executor.shutdown(wait=True)
        self.shutdown()
