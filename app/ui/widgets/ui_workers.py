import uuid
import queue
from functools import partial
from typing import TYPE_CHECKING, Dict
import traceback
import os
import threading
import subprocess
from datetime import datetime

import cv2
import torch
import numpy
from PySide6 import QtCore as qtc
from PySide6.QtGui import QPixmap
from PIL import Image

from app.processors.models_data import detection_model_mapping, landmark_model_mapping
from app.helpers import miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA, CAMERA_BACKENDS
from app.processors.workers.frame_worker import FrameWorker


if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

class TargetMediaLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, QPixmap, str, str)  # Signal with media path and QPixmap and file_type, media_id
    webcam_thumbnail_ready = qtc.Signal(str, QPixmap, str, str, int, int)
    finished = qtc.Signal()  # Signal to indicate completion

    def __init__(self, main_window: 'MainWindow', folder_name=False, files_list=None, media_ids=None, webcam_mode=False, parent=None,):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.media_ids = media_ids or []
        self.webcam_mode = webcam_mode
        self._running = True  # Flag to control the running state
        
        # Ensure thumbnail directory exists
        misc_helpers.ensure_thumbnail_dir()

    def run(self):
        if self.folder_name:
            self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list:
            self.load_videos_and_images_from_files_list(self.files_list)
        if self.webcam_mode:
            self.load_webcams()
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        # Initially hide the placeholder text
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        video_files = misc_helpers.get_video_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])
        image_files = misc_helpers.get_image_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])

        i=0
        media_files = video_files + image_files
        for media_file in media_files:
            if not self._running:  # Check if the thread is still running
                break
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
            i+=1
        # Show/Hide the placeholder text based on the number of items in ListWidget
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_videos_and_images_from_files_list(self, files_list):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        media_files = files_list
        i=0
        for media_file_path in media_files:
            if not self._running:  # Check if the thread is still running
                break
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type=file_type)
            if self.media_ids:
                media_id = self.media_ids[i]
            else:
                media_id = str(uuid.uuid1().int)
            if pixmap:
                # Emit the signal to update GUI
                self.thumbnail_ready.emit(media_file_path, pixmap, file_type,media_id)
            i+=1
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_webcams(self,):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        camera_backend = CAMERA_BACKENDS[self.main_window.control['WebcamBackendSelection']]
        for i in range(int(self.main_window.control['WebcamMaxNoSelection'])):
            try:
                pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path=f'Webcam {i}', file_type='webcam', webcam_index=i, webcam_backend=camera_backend)
                media_id = str(uuid.uuid1().int)

                if pixmap:
                    # Emit the signal to update GUI
                    self.webcam_thumbnail_ready.emit(f'Webcam {i}', pixmap, 'webcam',media_id, i, camera_backend)
            except Exception: # pylint: disable=broad-exception-caught
                traceback.print_exc()
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()

class InputFacesLoaderWorker(qtc.QThread):
    # Define signals to emit when loading is done or if there are updates
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QPixmap, str)
    finished = qtc.Signal()  # Signal to indicate completion
    def __init__(self, main_window: 'MainWindow', media_path=False, folder_name=False, files_list=None, face_ids=None,  parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.face_ids = face_ids or []
        self._running = True  # Flag to control the running state
        self.was_playing = True
        self.pre_load_detection_recognition_models()
        
    def pre_load_detection_recognition_models(self):
        control = self.main_window.control.copy()
        detect_model = detection_model_mapping[control['DetectorModelSelection']]
        landmark_detect_model = landmark_model_mapping[control['LandmarkDetectModelSelection']]
        models_processor = self.main_window.models_processor
        if self.main_window.video_processor.processing:
            was_playing = True
            self.main_window.buttonMediaPlay.click()
        else:
            was_playing = False
        if not models_processor.models[detect_model]:
            models_processor.models[detect_model] = models_processor.load_model(detect_model)
        if not models_processor.models[landmark_detect_model] and control['LandmarkDetectToggle']:
            models_processor.models[landmark_detect_model] = models_processor.load_model(landmark_detect_model)
        for recognition_model in ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace', 'CSCSIDArcFace']:
            if not models_processor.models[recognition_model]:
                models_processor.models[recognition_model] = models_processor.load_model(recognition_model)
        if was_playing:
            self.main_window.buttonMediaPlay.click()

    def run(self):
        if self.folder_name or self.files_list:
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, True)
            self.load_faces(self.folder_name, self.files_list)
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, False)

    def load_faces(self, folder_name=False, files_list=None):
        control = self.main_window.control.copy()
        files_list = files_list or []
        image_files = []
        if folder_name:
            image_files = misc_helpers.get_image_files(self.folder_name, self.main_window.control['InputFacesFolderRecursiveToggle'])
        elif files_list:
            image_files = files_list

        i=0
        image_files.sort()
        for image_file_path in image_files:
            if not self._running:  # Check if the thread is still running
                break
            if not misc_helpers.is_image_file(image_file_path):
                return
            if folder_name:
                image_file_path = os.path.join(folder_name, image_file_path)
            frame = misc_helpers.read_image_file(image_file_path)
            if frame is None:
                continue
            # Frame must be in RGB format
            frame = frame[..., ::-1]  # Swap the channels from BGR to RGB

            img = torch.from_numpy(frame.astype('uint8')).to(self.main_window.models_processor.device)
            img = img.permute(2,0,1)
            _, kpss_5, _ = self.main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=1, score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=control["DetectFromPointsToggle"], rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])

            # If atleast one face is found
            # found_face = []
            face_kps = False
            try:
                face_kps = kpss_5[0]
            except IndexError:
                continue
            if face_kps.any():
                face_emb, cropped_img = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                cropped_img = cropped_img.cpu().numpy()
                cropped_img = cropped_img[..., ::-1]  # Swap the channels from RGB to BGR
                face_img = numpy.ascontiguousarray(cropped_img)
                # crop = cv2.resize(face[2].cpu().numpy(), (82, 82))
                pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, face_img)

                embedding_store: Dict[str, numpy.ndarray] = {}
                # Ottenere i valori di 'options'
                options = SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
                for option in options:
                    if option != control['RecognitionModelSelection']:
                        target_emb, _ = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], option)
                        embedding_store[option] = target_emb
                    else:
                        embedding_store[control['RecognitionModelSelection']] = face_emb
                if not self.face_ids:
                    face_id = str(uuid.uuid1().int)
                else:
                    face_id = self.face_ids[i]
                self.thumbnail_ready.emit(image_file_path, face_img, embedding_store, pixmap, face_id)
                i+=1
        torch.cuda.empty_cache()
        self.finished.emit()

    def stop(self):
        """Stop the thread by setting the running flag to False."""
        self._running = False
        self.wait()

class FilterWorker(qtc.QThread):
    filtered_results = qtc.Signal(list)

    def __init__(self, main_window: 'MainWindow', search_text='', filter_list='target_videos'):
        super().__init__()
        self.main_window = main_window
        self.search_text = search_text
        self.filter_list = filter_list
        self.filter_list_widget = self.get_list_widget()
        self.filtered_results.connect(partial(filter_actions.update_filtered_list, main_window, self.filter_list_widget))

    def get_list_widget(self,):
        list_widget = False
        if self.filter_list == 'target_videos':
            list_widget = self.main_window.targetVideosList
        elif self.filter_list == 'input_faces':
            list_widget = self.main_window.inputFacesList
        elif self.filter_list == 'merged_embeddings':
            list_widget = self.main_window.inputEmbeddingsList
        return list_widget

    def run(self,):
        if self.filter_list == 'target_videos':
            self.filter_target_videos(self.main_window, self.search_text)
        elif self.filter_list == 'input_faces':
            self.filter_input_faces(self.main_window, self.search_text)
        elif self.filter_list == 'merged_embeddings':
            self.filter_merged_embeddings(self.main_window, self.search_text)


    def filter_target_videos(self, main_window: 'MainWindow', search_text: str = ''):
        search_text = main_window.targetVideosSearchBox.text().lower()
        include_file_types = []
        if main_window.filterImagesCheckBox.isChecked():
            include_file_types.append('image')
        if main_window.filterVideosCheckBox.isChecked():
            include_file_types.append('video')
        if main_window.filterWebcamsCheckBox.isChecked():
            include_file_types.append('webcam')

        visible_indices = []
        for i in range(main_window.targetVideosList.count()):
            item = main_window.targetVideosList.item(i)
            item_widget = main_window.targetVideosList.itemWidget(item)
            if ((not search_text or search_text in item_widget.media_path.lower()) and 
                (item_widget.file_type in include_file_types)):
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def filter_input_faces(self, main_window: 'MainWindow', search_text: str):
        search_text = search_text.lower()
        visible_indices = []

        for i in range(main_window.inputFacesList.count()):
            item = main_window.inputFacesList.item(i)
            item_widget = main_window.inputFacesList.itemWidget(item)
            if not search_text or search_text in item_widget.media_path.lower():
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def filter_merged_embeddings(self, main_window: 'MainWindow', search_text: str):
        search_text = search_text.lower()
        visible_indices = []

        for i in range(main_window.inputEmbeddingsList.count()):
            item = main_window.inputEmbeddingsList.item(i)
            item_widget = main_window.inputEmbeddingsList.itemWidget(item)
            if not search_text or search_text in item_widget.embedding_name.lower():
                visible_indices.append(i)

        self.filtered_results.emit(visible_indices)

    def stop_thread(self):
        self.quit()
        self.wait()

class BatchProcessorWorker(qtc.QThread):
    """
    Worker thread to process a batch of images and videos with the current settings.
    This prevents the main UI from freezing during a potentially long operation.
    """
    progress = qtc.Signal(int, int, str, int, int) # file_idx, total_files, filename, frame_idx, total_frames
    output_directory_updated = qtc.Signal(str)
    finished = qtc.Signal(str)
    _lock = threading.Lock()

    def __init__(self, main_window: 'MainWindow', media_paths: list, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.media_paths = media_paths
        self._is_running = True

    def run(self):
        if not BatchProcessorWorker._lock.acquire(blocking=False):
            print("Batch processing skipped: another process is already running.")
            self.finished.emit("Another batch process is already running.")
            return

        try:
            total_media = len(self.media_paths)
            processed_count = 0

            for i, media_path in enumerate(self.media_paths):
                if not self._is_running:
                    break

                if misc_helpers.is_video_file(media_path):
                    self._process_video(i, total_media, media_path)
                elif misc_helpers.is_image_file(media_path):
                    self._process_image(i, total_media, media_path)
                else:
                    print(f"Skipping unsupported file type: {media_path}")
                
                processed_count = i + 1

            if self._is_running:
                self.finished.emit(f"Successfully processed {processed_count} of {total_media} items.")
            else:
                self.finished.emit(f"Batch processing was cancelled by the user after {processed_count} items.")

        except Exception as e:
            traceback.print_exc()
            self.finished.emit(f"An error occurred during batch processing: {e}")
        finally:
            BatchProcessorWorker._lock.release()
            torch.cuda.empty_cache()

    def _process_image(self, file_idx, total_files, image_path):
        self.progress.emit(file_idx, total_files, os.path.basename(image_path), 0, 0)
        
        output_to_target = self.main_window.control.get('OutputToTargetLocationToggle', False)
        if output_to_target:
            output_folder = os.path.dirname(image_path)
        else:
            output_folder = self.main_window.control.get('OutputMediaFolder')
        
        self.output_directory_updated.emit(output_folder)

        frame_bgr = misc_helpers.read_image_file(image_path)
        if frame_bgr is None:
            print(f"Warning: Could not read image {image_path}. Skipping.")
            return

        dummy_queue = queue.Queue()
        dummy_queue.put(0) 
        worker = FrameWorker(
            frame=frame_bgr[..., ::-1].copy(), # to RGB
            main_window=self.main_window,
            frame_number=0,
            frame_queue=dummy_queue,
            is_single_frame=True,
            is_batch_processing=True
        )
        worker.run()
        processed_frame_bgr = worker.frame

        output_path = misc_helpers.get_output_file_path(image_path, output_folder, 'image')
        # Change the extension to .jpg for saving
        base, _ = os.path.splitext(output_path)
        save_filename = base + '.jpg'
        
        try:
            # The frame from the processor is BGR, so we need to convert it to RGB for PIL
            pil_image = Image.fromarray(processed_frame_bgr[..., ::-1])
            
            # Save as JPEG with high quality.
            pil_image.save(save_filename, 'JPEG', quality=85)
        except Exception as e:
            print(f"ERROR: Could not save processed image to {save_filename}. Reason: {e}")

        self.progress.emit(file_idx + 1, total_files, os.path.basename(image_path), 1, 1)

    def _process_video(self, file_idx, total_files, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file {video_path}. Skipping.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Determine output folder
        output_to_target = self.main_window.control.get('OutputToTargetLocationToggle', False)
        if output_to_target:
            output_folder = os.path.dirname(video_path)
        else:
            output_folder = self.main_window.control.get('OutputMediaFolder')

        self.output_directory_updated.emit(output_folder)

        # Setup FFMPEG subprocess
        date_and_time = datetime.now().strftime(r'%Y_%m_%d_%H_%M_%S')
        temp_file = os.path.join(output_folder, f'temp_output_{date_and_time}.mp4')
        final_file_path = misc_helpers.get_output_file_path(video_path, output_folder, 'video')

        ffmpeg_args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps),
            "-i", "pipe:",
            "-vf", f"pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p",
            "-c:v", "libx264", "-crf", "18", temp_file
        ]
        
        recording_sp = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        dummy_queue = queue.Queue()

        frame_count = 0
        while self._is_running:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            self.progress.emit(file_idx, total_files, os.path.basename(video_path), frame_count, total_frames)

            dummy_queue.put(frame_count)
            worker = FrameWorker(
                frame=frame_bgr[..., ::-1].copy(), # to RGB
                main_window=self.main_window,
                frame_number=frame_count,
                frame_queue=dummy_queue,
                is_single_frame=True,
                is_batch_processing=True
            )
            worker.run()
            
            recording_sp.stdin.write(worker.frame.tobytes())
            frame_count += 1
        
        # Finalize video processing
        cap.release()
        recording_sp.stdin.close()
        recording_sp.wait()
        
        # Add audio and cleanup
        if self._is_running:
            print("Adding audio to processed video...")
            audio_args = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-i", temp_file, "-i", video_path,
                "-c", "copy", "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest", final_file_path
            ]
            subprocess.run(audio_args, check=False)
        
        if os.path.exists(temp_file):
            os.remove(temp_file)

        self.progress.emit(file_idx + 1, total_files, os.path.basename(video_path), total_frames, total_frames)


    def stop(self):
        """Stops the worker thread safely."""
        self._is_running = False