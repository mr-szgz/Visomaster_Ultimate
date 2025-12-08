import uuid
import queue
from functools import partial
from typing import TYPE_CHECKING, Dict
import traceback
import os
import threading
import subprocess
from datetime import datetime
import platform

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
    thumbnail_ready = qtc.Signal(str, QPixmap, str, str)
    webcam_thumbnail_ready = qtc.Signal(str, QPixmap, str, str, int, int)
    finished = qtc.Signal()

    def __init__(self, main_window: 'MainWindow', folder_name=False, files_list=None, media_ids=None, webcam_mode=False, parent=None,):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.media_ids = media_ids or []
        self.webcam_mode = webcam_mode
        self._running = True
        misc_helpers.ensure_thumbnail_dir()

    def run(self):
        if self.folder_name: self.load_videos_and_images_from_folder(self.folder_name)
        if self.files_list: self.load_videos_and_images_from_files_list(self.files_list)
        if self.webcam_mode: self.load_webcams()
        self.finished.emit()

    def load_videos_and_images_from_folder(self, folder_name):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        video_files = misc_helpers.get_video_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])
        image_files = misc_helpers.get_image_files(folder_name, self.main_window.control['TargetMediaFolderRecursiveToggle'])
        media_files = video_files + image_files
        for i, media_file in enumerate(media_files):
            if not self._running: break
            media_file_path = os.path.join(folder_name, media_file)
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type)
            media_id = self.media_ids[i] if self.media_ids else str(uuid.uuid1().int)
            if pixmap: self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_videos_and_images_from_files_list(self, files_list):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        for i, media_file_path in enumerate(files_list):
            if not self._running: break
            file_type = misc_helpers.get_file_type(media_file_path)
            pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path, file_type=file_type)
            media_id = self.media_ids[i] if self.media_ids else str(uuid.uuid1().int)
            if pixmap: self.thumbnail_ready.emit(media_file_path, pixmap, file_type, media_id)
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def load_webcams(self,):
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, True)
        camera_backend = CAMERA_BACKENDS[self.main_window.control['WebcamBackendSelection']]
        for i in range(int(self.main_window.control['WebcamMaxNoSelection'])):
            try:
                pixmap = common_widget_actions.extract_frame_as_pixmap(media_file_path=f'Webcam {i}', file_type='webcam', webcam_index=i, webcam_backend=camera_backend)
                media_id = str(uuid.uuid1().int)
                if pixmap: self.webcam_thumbnail_ready.emit(f'Webcam {i}', pixmap, 'webcam', media_id, i, camera_backend)
            except Exception: traceback.print_exc()
        self.main_window.placeholder_update_signal.emit(self.main_window.targetVideosList, False)

    def stop(self):
        self._running = False
        self.wait()

class InputFacesLoaderWorker(qtc.QThread):
    thumbnail_ready = qtc.Signal(str, numpy.ndarray, object, QPixmap, str)
    finished = qtc.Signal()

    def __init__(self, main_window: 'MainWindow', media_path=False, folder_name=False, files_list=None, face_ids=None,  parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.folder_name = folder_name
        self.files_list = files_list or []
        self.face_ids = face_ids or []
        self._running = True
        self.pre_load_detection_recognition_models()
        
    def pre_load_detection_recognition_models(self):
        control = self.main_window.control.copy()
        models_processor = self.main_window.models_processor
        detect_model = detection_model_mapping[control['DetectorModelSelection']]
        landmark_detect_model = landmark_model_mapping[control['LandmarkDetectModelSelection']]
        was_playing = self.main_window.video_processor.processing
        if was_playing: self.main_window.buttonMediaPlay.click()
        if not models_processor.models.get(detect_model): models_processor.models[detect_model] = models_processor.load_model(detect_model)
        if control['LandmarkDetectToggle'] and not models_processor.models.get(landmark_detect_model): models_processor.models[landmark_detect_model] = models_processor.load_model(landmark_detect_model)
        for rec_model in ['Inswapper128ArcFace', 'SimSwapArcFace', 'GhostArcFace', 'CSCSArcFace', 'CSCSIDArcFace']:
            if not models_processor.models.get(rec_model): models_processor.models[rec_model] = models_processor.load_model(rec_model)
        if was_playing: self.main_window.buttonMediaPlay.click()

    def run(self):
        if self.folder_name or self.files_list:
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, True)
            self.load_faces(self.folder_name, self.files_list)
            self.main_window.placeholder_update_signal.emit(self.main_window.inputFacesList, False)

    def load_faces(self, folder_name=False, files_list=None):
        control = self.main_window.control.copy()
        if folder_name: image_files = misc_helpers.get_image_files(self.folder_name, self.main_window.control['InputFacesFolderRecursiveToggle'])
        else: image_files = files_list or []
        image_files.sort()
        for i, image_file in enumerate(image_files):
            if not self._running: break
            image_file_path = os.path.join(folder_name, image_file) if folder_name else image_file
            if not misc_helpers.is_image_file(image_file_path): continue
            frame = misc_helpers.read_image_file(image_file_path)
            if frame is None: continue
            frame_rgb = frame[..., ::-1]
            img = torch.from_numpy(frame_rgb.astype('uint8')).to(self.main_window.models_processor.device).permute(2,0,1)
            _, kpss_5, _ = self.main_window.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=1, score=control['DetectorScoreSlider']/100.0, use_landmark_detection=control['LandmarkDetectToggle'], landmark_detect_mode=control['LandmarkDetectModelSelection'], landmark_score=control["LandmarkDetectScoreSlider"]/100.0)
            try: face_kps = kpss_5[0]
            except IndexError: continue
            if face_kps.any():
                face_emb, cropped_img_tensor = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                cropped_img = numpy.ascontiguousarray(cropped_img_tensor.cpu().numpy()[..., ::-1])
                pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, cropped_img)
                embedding_store = {}
                options = SETTINGS_LAYOUT_DATA['Face Recognition']['RecognitionModelSelection']['options']
                for option in options:
                    if option != control['RecognitionModelSelection']:
                        target_emb, _ = self.main_window.models_processor.run_recognize_direct(img, face_kps, control['SimilarityTypeSelection'], option)
                        embedding_store[option] = target_emb
                    else: embedding_store[control['RecognitionModelSelection']] = face_emb
                face_id = self.face_ids[i] if self.face_ids and i < len(self.face_ids) else str(uuid.uuid1().int)
                self.thumbnail_ready.emit(image_file_path, cropped_img, embedding_store, pixmap, face_id)
        torch.cuda.empty_cache()
        self.finished.emit()

    def stop(self):
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
        if self.filter_list == 'target_videos': return self.main_window.targetVideosList
        if self.filter_list == 'input_faces': return self.main_window.inputFacesList
        if self.filter_list == 'merged_embeddings': return self.main_window.inputEmbeddingsList
        return None

    def run(self,):
        if self.filter_list == 'target_videos': self.filter_target_videos()
        elif self.filter_list == 'input_faces': self.filter_input_faces()
        elif self.filter_list == 'merged_embeddings': self.filter_merged_embeddings()

    def filter_target_videos(self):
        search_text = self.main_window.targetVideosSearchBox.text().lower()
        include_file_types = []
        if self.main_window.filterImagesCheckBox.isChecked(): include_file_types.append('image')
        if self.main_window.filterVideosCheckBox.isChecked(): include_file_types.append('video')
        if self.main_window.filterWebcamsCheckBox.isChecked(): include_file_types.append('webcam')
        visible_indices = [i for i in range(self.main_window.targetVideosList.count())
                           if (not search_text or search_text in self.main_window.targetVideosList.itemWidget(self.main_window.targetVideosList.item(i)).media_path.lower()) and
                              (self.main_window.targetVideosList.itemWidget(self.main_window.targetVideosList.item(i)).file_type in include_file_types)]
        self.filtered_results.emit(visible_indices)

    def filter_input_faces(self):
        search_text = self.main_window.inputFacesSearchBox.text().lower()
        visible_indices = [i for i in range(self.main_window.inputFacesList.count())
                           if not search_text or search_text in self.main_window.inputFacesList.itemWidget(self.main_window.inputFacesList.item(i)).media_path.lower()]
        self.filtered_results.emit(visible_indices)

    def filter_merged_embeddings(self):
        search_text = self.main_window.inputEmbeddingsSearchBox.text().lower()
        visible_indices = [i for i in range(self.main_window.inputEmbeddingsList.count())
                           if not search_text or search_text in self.main_window.inputEmbeddingsList.itemWidget(self.main_window.inputEmbeddingsList.item(i)).embedding_name.lower()]
        self.filtered_results.emit(visible_indices)

    def stop_thread(self):
        self.quit()
        self.wait()

class BatchProcessorWorker(qtc.QThread):
    progress = qtc.Signal(int, int, str, int, int)
    output_directory_updated = qtc.Signal(str)
    finished = qtc.Signal(str)
    _lock = threading.Lock()

    def __init__(self, main_window: 'MainWindow', media_paths: list, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.media_paths = media_paths
        self._is_running = True

    def _select_single_embedding(self, embeddings_dict):
        try:
            if embeddings_dict is None: return None
            if isinstance(embeddings_dict, dict) and "embedding_store" in embeddings_dict:
                embeddings_dict = embeddings_dict["embedding_store"]
            if isinstance(embeddings_dict, dict):
                model_name = self.main_window.control.get('RecognitionModelSelection')
                emb = embeddings_dict.get(model_name)
                if emb is not None:
                    import numpy as _np
                    return _np.asarray(emb, dtype=_np.float32)
        except Exception: traceback.print_exc()
        return None

    def run(self):
        if not BatchProcessorWorker._lock.acquire(blocking=False):
            self.finished.emit("Another batch process is already running.")
            return

        try:
            process_for_each_source = self.main_window.control.get('ProcessForEachSourceImageToggle', False)
            process_for_each_embedding = self.main_window.control.get('ProcessForEachEmbeddingToggle', False)
            
            num_sources_to_process = 0
            if process_for_each_source: num_sources_to_process += self.main_window.inputFacesList.count()
            if process_for_each_embedding: num_sources_to_process += self.main_window.inputEmbeddingsList.count()
            if num_sources_to_process == 0: num_sources_to_process = 1

            num_targets = len(self.media_paths)
            total_operations = num_sources_to_process * num_targets
            
            RESERVED_NAMES = set()
            if platform.system() == "Windows": RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9"}

            def sanitize_name(name):
                sanitized = "".join([c for c in name if c.isalnum() or c in '._-']).rstrip()
                if not sanitized: return "unnamed_source"
                if sanitized.upper() in RESERVED_NAMES: return f"{sanitized}_"
                return sanitized

            source_iterations = []
            if process_for_each_source:
                for i in range(self.main_window.inputFacesList.count()):
                    widget = self.main_window.inputFacesList.itemWidget(self.main_window.inputFacesList.item(i))
                    source_iterations.append({'name': sanitize_name(os.path.splitext(os.path.basename(widget.media_path))[0]), 'embeddings': widget.embedding_store})
            if process_for_each_embedding:
                for i in range(self.main_window.inputEmbeddingsList.count()):
                    widget = self.main_window.inputEmbeddingsList.itemWidget(self.main_window.inputEmbeddingsList.item(i))
                    source_iterations.append({'name': sanitize_name(widget.embedding_name), 'embeddings': widget.embedding_store})
            if not source_iterations: source_iterations.append({'name': 'default', 'embeddings': None})

            operation_count = 0
            total_outputs = 0
            for media_path in self.media_paths:
                if not self._is_running: break
                for source_iter in source_iterations:
                    if not self._is_running: break
                    operation_count += 1
                    self.main_window.batch_processing_source_embedding_override = source_iter['embeddings']
                    source_name = source_iter['name'] if source_iter['name'] != 'default' else None
                    progress_label = f"({operation_count}/{total_operations}) {os.path.basename(media_path)}"
                    if source_name: progress_label += f" with {source_name}"
                    if misc_helpers.is_video_file(media_path): self._process_video(operation_count, total_operations, media_path, source_name, progress_label)
                    elif misc_helpers.is_image_file(media_path): self._process_image(operation_count, total_operations, media_path, source_name, progress_label)
                    if self._is_running: total_outputs += 1
            
            msg = f"Successfully generated {total_outputs} outputs." if self._is_running else f"Cancelled after generating {total_outputs} outputs."
            self.finished.emit(msg)

        except Exception as e:
            traceback.print_exc()
            self.finished.emit(f"An error occurred: {e}")
        finally:
            if hasattr(self.main_window, 'batch_processing_source_embedding_override'): del self.main_window.batch_processing_source_embedding_override
            BatchProcessorWorker._lock.release()
            torch.cuda.empty_cache()

    def _get_output_folder(self, base_media_path, source_name):
        try:
            if self.main_window.control.get('OutputToTargetLocationToggle', False): base_folder = os.path.dirname(base_media_path)
            else: base_folder = self.main_window.control.get('OutputMediaFolder')
            if not base_folder or not os.path.isdir(base_folder): raise IOError(f"Base output directory is invalid: {base_folder}")
            final_folder = os.path.join(base_folder, source_name) if self.main_window.control.get('ClusterOutputBySourceToggle', True) and source_name else base_folder
            os.makedirs(final_folder, exist_ok=True)
            return final_folder
        except Exception as e:
            print(f"[Batch][Path] FATAL ERROR creating output directory. Reason: {e}")
            raise

    def _process_image(self, current_op, total_ops, image_path, source_name, progress_label):
        self.progress.emit(current_op - 1, total_ops, progress_label, 0, 0)
        output_folder = self._get_output_folder(image_path, source_name)
        self.output_directory_updated.emit(output_folder)
        frame_bgr = misc_helpers.read_image_file(image_path)
        if frame_bgr is None:
            self.progress.emit(current_op, total_ops, f"{progress_label} (Skipped)", 1, 1)
            return
        single_emb = self._select_single_embedding(getattr(self.main_window, 'batch_processing_source_embedding_override', None))
        dummy_queue = queue.Queue()
        dummy_queue.put(0)
        worker = FrameWorker(frame=frame_bgr[..., ::-1].copy(), main_window=self.main_window, frame_number=0, frame_queue=dummy_queue, is_single_frame=True, is_batch_processing=True, source_embedding_override=single_emb)
        worker.run()
        processed_frame_bgr = worker.frame
        original_stem = os.path.splitext(os.path.basename(image_path))[0]
        output_stem = f"{original_stem}_swapped" + (f"_{source_name}" if source_name else "")
        save_filename = os.path.join(output_folder, output_stem + '.jpg')
        Image.fromarray(processed_frame_bgr[..., ::-1]).save(save_filename, 'JPEG', quality=85)
        self.progress.emit(current_op, total_ops, f"{progress_label} (Done)", 1, 1)

    def _process_video(self, current_op, total_ops, video_path, source_name, progress_label):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.progress.emit(current_op, total_ops, f"{progress_label} (Skipped)", 1, 1)
            return
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_folder = self._get_output_folder(video_path, source_name)
        self.output_directory_updated.emit(output_folder)
        original_stem = os.path.splitext(os.path.basename(video_path))[0]
        output_stem = f"{original_stem}_swapped" + (f"_{source_name}" if source_name else "")
        final_file_path = os.path.join(output_folder, output_stem + '.mp4')
        temp_file = os.path.join(output_folder, f"temp_{output_stem}_{datetime.now().strftime(r'%Y%m%d_%H%M%S')}.mp4")
        ffmpeg_args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}", "-r", str(cap.get(cv2.CAP_PROP_FPS)), "-i", "pipe:", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,format=yuvj420p", "-c:v", "libx264", "-crf", "18", temp_file]
        recording_sp = subprocess.Popen(ffmpeg_args, stdin=subprocess.PIPE)
        dummy_queue = queue.Queue()
        frame_count = 0
        while self._is_running:
            ret, frame_bgr = cap.read()
            if not ret: break
            self.progress.emit(current_op - 1, total_ops, progress_label, frame_count, total_frames)
            single_emb = self._select_single_embedding(getattr(self.main_window, 'batch_processing_source_embedding_override', None))
            dummy_queue.put(frame_count)
            worker = FrameWorker(frame=frame_bgr[..., ::-1].copy(), main_window=self.main_window, frame_number=frame_count, frame_queue=dummy_queue, is_single_frame=True, is_batch_processing=True, source_embedding_override=single_emb)
            worker.run()
            recording_sp.stdin.write(worker.frame.tobytes())
            frame_count += 1
        cap.release()
        recording_sp.stdin.close()
        recording_sp.wait()
        if self._is_running:
            audio_args = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", temp_file, "-i", video_path, "-c", "copy", "-map", "0:v:0", "-map", "1:a:0?", "-shortest", final_file_path]
            subprocess.run(audio_args, check=False)
        if os.path.exists(temp_file): os.remove(temp_file)
        self.progress.emit(current_op, total_ops, f"{progress_label} (Done)", total_frames, total_frames)

    def stop(self):
        self._is_running = False

class ClipboardBatchSaveFrameWorker(qtc.QThread):
    progress = qtc.Signal(int, int, str, int, int)
    finished = qtc.Signal(str)
    output_directory_updated = qtc.Signal(str)

    def __init__(self, main_window: 'MainWindow', media_paths: list, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.media_paths = media_paths
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        merged_embeddings = getattr(self.main_window, 'merged_embeddings', {})
        if not merged_embeddings:
            self.finished.emit("No embeddings found to process against.")
            return

        total_operations = len(self.media_paths) * len(merged_embeddings)
        operation_count = 0
        total_outputs = 0
        for media_path in self.media_paths:
            if not self._is_running: break
            frame_bgr = None
            if misc_helpers.is_video_file(media_path):
                cap = cv2.VideoCapture(media_path)
                if cap.isOpened(): ret, frame_bgr = cap.read(); cap.release()
            elif misc_helpers.is_image_file(media_path): frame_bgr = misc_helpers.read_image_file(media_path)
            if frame_bgr is None: continue
            frame_rgb = frame_bgr[..., ::-1].copy()
            for emb_id, emb_btn in merged_embeddings.items():
                if not self._is_running: break
                operation_count += 1
                emb_name = getattr(emb_btn, 'embedding_name', str(emb_id))
                progress_label = f"({operation_count}/{total_operations}) {os.path.basename(media_path)} with {emb_name}"
                self.progress.emit(operation_count - 1, total_operations, progress_label, 0, 0)
                model_name = self.main_window.control.get('RecognitionModelSelection')
                single_emb = numpy.asarray(emb_btn.embedding_store.get(model_name), dtype=numpy.float32)
                dummy_queue = queue.Queue(); dummy_queue.put(0)
                worker = FrameWorker(frame=frame_rgb.copy(), main_window=self.main_window, frame_number=0, frame_queue=dummy_queue, is_single_frame=True, is_batch_processing=True, source_embedding_override=single_emb)
                worker.run()
                self._save_frame(worker.frame, media_path, emb_name)
                total_outputs += 1
                self.progress.emit(operation_count, total_operations, progress_label, 1, 1)

        msg = f"Batch Save Frame complete. Saved {total_outputs} images." if self._is_running else "Batch Save Frame cancelled."
        self.finished.emit(msg)
    
    def _save_frame(self, frame_bgr, original_media_path, source_name):
        base_output_folder = os.path.dirname(original_media_path) if self.main_window.control.get('OutputToTargetLocationToggle', False) else self.main_window.control.get('OutputMediaFolder')
        if not base_output_folder: return
        output_folder = base_output_folder
        if self.main_window.control.get('ClusterOutputBySourceToggle', True) and source_name:
            sanitized_name = "".join([c for c in source_name if c.isalnum() or c in "._-"]).rstrip()
            output_folder = os.path.join(base_output_folder, sanitized_name)
        try:
            os.makedirs(output_folder, exist_ok=True)
            self.output_directory_updated.emit(output_folder)
        except Exception as e:
            print(f"Error creating directory {output_folder}: {e}")
            return
        stem = os.path.splitext(os.path.basename(original_media_path))[0]
        if self.main_window.control.get('ClusterOutputBySourceToggle', True) and source_name: stem = f"{stem}_{source_name}"
        ts = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        rand = ''.join(__import__('secrets').choice(__import__('string').ascii_letters + __import__('string').digits) for _ in range(6))
        save_filename = os.path.join(output_folder, f"{ts}_{stem}_{rand}.jpg")
        try: Image.fromarray(frame_bgr[..., ::-1]).save(save_filename, 'JPEG', quality=85)
        except Exception as e: print(f"Error saving image {save_filename}: {e}")