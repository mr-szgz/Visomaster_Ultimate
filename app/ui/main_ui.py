import os
from typing import Dict, List
from pathlib import Path
from functools import partial
import copy

from PySide6 import QtWidgets, QtGui
from PySide6 import QtCore
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices

from app.ui.core.main_window import Ui_MainWindow
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import card_actions
from app.ui.widgets.actions import layout_actions
from app.ui.widgets.actions import video_control_actions
from app.ui.widgets.actions import filter_actions
from app.ui.widgets.actions import save_load_actions
from app.ui.widgets.actions import list_view_actions
from app.ui.widgets.actions import graphics_view_actions
from app.ui.widgets.advanced_embedding_editor import EmbeddingGUI

from app.processors.video_processor import VideoProcessor
from app.processors.models_processor import ModelsProcessor
from app.ui.widgets import widget_components
from app.ui.widgets.event_filters import GraphicsViewEventFilter, VideoSeekSliderEventFilter, videoSeekSliderLineEditEventFilter, ListWidgetEventFilter
from app.ui.widgets import ui_workers
from app.ui.widgets.common_layout_data import COMMON_LAYOUT_DATA
from app.ui.widgets.swapper_layout_data import SWAPPER_LAYOUT_DATA
from app.ui.widgets.settings_layout_data import SETTINGS_LAYOUT_DATA
from app.ui.widgets.face_editor_layout_data import FACE_EDITOR_LAYOUT_DATA
from app.helpers.miscellaneous import DFM_MODELS_DATA, ParametersDict
from app.helpers.typing_helper import FacesParametersTypes, ParametersTypes, ControlTypes, MarkerTypes
import app.helpers.miscellaneous as misc_helpers

ParametersWidgetTypes = Dict[str, widget_components.ToggleButton|widget_components.SelectionBox|widget_components.ParameterDecimalSlider|widget_components.ParameterSlider|widget_components.ParameterText]

class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    placeholder_update_signal = QtCore.Signal(QtWidgets.QListWidget, bool)
    gpu_memory_update_signal = QtCore.Signal(int, int)
    model_loading_signal = QtCore.Signal()
    model_loaded_signal = QtCore.Signal()
    display_messagebox_signal = QtCore.Signal(str, str, QtWidgets.QWidget)

    def initialize_variables(self):
        self.video_loader_worker: ui_workers.TargetMediaLoaderWorker | None = None
        self.input_faces_loader_worker: ui_workers.InputFacesLoaderWorker | None = None
        self.batch_worker: ui_workers.BatchProcessorWorker | None = None
        self.clipboard_batch_save_worker: ui_workers.ClipboardBatchSaveFrameWorker | None = None
        self.target_videos_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='target_videos')
        self.input_faces_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='input_faces')
        self.merged_embeddings_filter_worker = ui_workers.FilterWorker(main_window=self, search_text='', filter_list='merged_embeddings')
        self.video_processor = VideoProcessor(self)
        self.models_processor = ModelsProcessor(self)
        self.target_videos: Dict[int, widget_components.TargetMediaCardButton] = {}
        self.target_faces: Dict[int, widget_components.TargetFaceCardButton] = {}
        self.input_faces: Dict[int, widget_components.InputFaceCardButton] = {}
        self.merged_embeddings: Dict[int, widget_components.EmbeddingCardButton] = {}
        self.cur_selected_target_face_button: widget_components.TargetFaceCardButton = False
        self.selected_video_button: widget_components.TargetMediaCardButton = False
        self.selected_target_face_id = False
        self.parameters: FacesParametersTypes = {} 
        self.default_parameters: ParametersTypes = {}
        self.copied_parameters: ParametersTypes = {}
        self.current_widget_parameters: ParametersTypes = {}
        self.markers: MarkerTypes = {}
        self.parameters_list = {}
        self.control: ControlTypes = {}
        self.parameter_widgets: ParametersWidgetTypes = {}
        self.loaded_embedding_filename: str = ''
        self.last_target_media_folder_path = ''
        self.last_input_media_folder_path = ''
        self.last_output_directory = ''
        self.embedding_sort_mode = "Off" # Can be "Off", "A-Z", "Z-A"
        self.unsorted_embedding_ids = []
        self.clipboard_timer = QTimer(self)
        self.last_clipboard_content = ""
        self.clipboard_file_queue: List[str] = []
        self.is_full_screen = False
        self.dfm_models_data = DFM_MODELS_DATA
        self.loading_new_media = False
        self.process_batch_on_load_finish = False
        self.process_batch_save_frame_on_load_finish = False
        self.files_for_batch_save_frame = []
        self.items_in_current_batch = []
        self.embedding_editor_window = None

        # --- FIX: Add a lock to prevent concurrent batch processes ---
        self.is_batch_processing_active = False

        self.gpu_memory_update_signal.connect(partial(common_widget_actions.set_gpu_memory_progressbar_value, self))
        self.placeholder_update_signal.connect(partial(common_widget_actions.update_placeholder_visibility, self))
        self.model_loading_signal.connect(partial(common_widget_actions.show_model_loading_dialog, self))
        self.model_loaded_signal.connect(partial(common_widget_actions.hide_model_loading_dialog, self))
        self.display_messagebox_signal.connect(partial(common_widget_actions.create_and_show_messagebox, self))

    def initialize_widgets(self):
        self.targetVideosList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.targetVideosList.setWrapping(True)
        self.targetVideosList.setResizeMode(QtWidgets.QListWidget.Adjust)
        self.inputFacesList.setFlow(QtWidgets.QListWidget.LeftToRight)
        self.inputFacesList.setWrapping(True)
        self.inputFacesList.setResizeMode(QtWidgets.QListWidget.Adjust)
        layout_actions.set_up_menu_actions(self)
        list_view_actions.set_up_list_widget_placeholder(self, self.targetVideosList)
        list_view_actions.set_up_list_widget_placeholder(self, self.inputFacesList)
        self.targetVideosList.setAcceptDrops(True)
        self.targetVideosList.viewport().setAcceptDrops(False)
        self.inputFacesList.setAcceptDrops(True)
        self.inputFacesList.viewport().setAcceptDrops(False)
        list_widget_event_filter = ListWidgetEventFilter(self, self)
        self.targetVideosList.installEventFilter(list_widget_event_filter)
        self.targetVideosList.viewport().installEventFilter(list_widget_event_filter)
        self.inputFacesList.installEventFilter(list_widget_event_filter)
        self.inputFacesList.viewport().installEventFilter(list_widget_event_filter)
        self.buttonTargetVideosPath.clicked.connect(partial(list_view_actions.select_target_medias, self, 'folder'))
        self.buttonInputFacesPath.clicked.connect(partial(list_view_actions.select_input_face_images, self, 'folder'))
        self.scene = QtWidgets.QGraphicsScene()
        self.graphicsViewFrame.setScene(self.scene)
        graphics_event_filter = GraphicsViewEventFilter(self, self.graphicsViewFrame,)
        self.graphicsViewFrame.installEventFilter(graphics_event_filter)
        video_control_actions.enable_zoom_and_pan(self.graphicsViewFrame)
        video_slider_event_filter = VideoSeekSliderEventFilter(self, self.videoSeekSlider)
        self.videoSeekSlider.installEventFilter(video_slider_event_filter)
        self.videoSeekSlider.valueChanged.connect(partial(video_control_actions.on_change_video_seek_slider, self))
        self.videoSeekSlider.sliderPressed.connect(partial(video_control_actions.on_slider_pressed, self))
        self.videoSeekSlider.sliderReleased.connect(partial(video_control_actions.on_slider_released, self))
        video_control_actions.set_up_video_seek_slider(self)
        self.frameAdvanceButton.clicked.connect(partial(video_control_actions.advance_video_slider_by_n_frames, self))
        self.frameRewindButton.clicked.connect(partial(video_control_actions.rewind_video_slider_by_n_frames, self))
        self.addMarkerButton.clicked.connect(partial(video_control_actions.add_video_slider_marker, self))
        self.removeMarkerButton.clicked.connect(partial(video_control_actions.remove_video_slider_marker, self))
        self.nextMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_next_nearest_marker, self))
        self.previousMarkerButton.clicked.connect(partial(video_control_actions.move_slider_to_previous_nearest_marker, self))
        self.viewFullScreenButton.clicked.connect(partial(video_control_actions.view_fullscreen, self))
        video_control_actions.set_up_video_seek_line_edit(self)
        video_seek_line_edit_event_filter = videoSeekSliderLineEditEventFilter(self, self.videoSeekLineEdit)
        self.videoSeekLineEdit.installEventFilter(video_seek_line_edit_event_filter)
        self.buttonMediaPlay.toggled.connect(partial(video_control_actions.play_video, self))
        self.buttonMediaRecord.toggled.connect(partial(video_control_actions.record_video, self))
        self.findTargetFacesButton.clicked.connect(partial(card_actions.find_target_faces, self))
        self.clearTargetFacesButton.clicked.connect(partial(card_actions.clear_target_faces, self))
        self.targetVideosSearchBox.textChanged.connect(partial(filter_actions.filter_target_videos, self))
        self.filterImagesCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterVideosCheckBox.clicked.connect(partial(filter_actions.filter_target_videos, self))
        self.filterWebcamsCheckBox.clicked.connect(partial(list_view_actions.load_target_webcams, self))
        self.inputFacesSearchBox.textChanged.connect(partial(filter_actions.filter_input_faces, self))
        self.inputEmbeddingsSearchBox.textChanged.connect(partial(filter_actions.filter_merged_embeddings, self))
        self.sortEmbeddingsButton.setText(f"Sorting: {self.embedding_sort_mode}")
        self.sortEmbeddingsButton.clicked.connect(self.toggle_embedding_sort)
        self.advancedEmbeddingEditorButton.clicked.connect(self.open_embedding_editor)
        self.openEmbeddingButton.clicked.connect(partial(save_load_actions.open_embeddings_from_file, self))
        self.saveEmbeddingButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self))
        self.saveEmbeddingAsButton.clicked.connect(partial(save_load_actions.save_embeddings_to_file, self, True))
        self.swapfacesButton.clicked.connect(partial(video_control_actions.process_swap_faces, self))
        self.editFacesButton.clicked.connect(partial(video_control_actions.process_edit_faces, self))
        self.saveImageButton.clicked.connect(partial(video_control_actions.save_current_frame_to_file, self))

        self.batchSaveFrameButton = QtWidgets.QPushButton(self.facesPanelGroupBox)
        self.batchSaveFrameButton.setObjectName("batchSaveFrameButton")
        self.batchSaveFrameButton.setFlat(True)
        try:
            self.horizontalLayout_4.insertWidget(self.horizontalLayout_4.indexOf(self.saveImageButton) + 1, self.batchSaveFrameButton)
        except Exception:
            self.horizontalLayout_4.addWidget(self.batchSaveFrameButton)
        self.batchSaveFrameButton.setText("Batch Save Single Frame")
        try:
            self.batchSaveFrameButton.setStyleSheet(self.saveImageButton.styleSheet())
        except Exception: pass
        self.batchSaveFrameButton.clicked.connect(partial(video_control_actions.batch_save_current_frame_all_embeddings, self))

        self.processAllImagesButton.clicked.connect(self.start_batch_processing)
        self.openOutputFolderButton.clicked.connect(self.open_last_output_directory)
        self.clearMemoryButton.clicked.connect(partial(common_widget_actions.clear_gpu_memory, self))
        self.parametersPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_parameters_panel, self))
        self.facesPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_faces_panel, self))
        self.mediaPanelCheckBox.toggled.connect(partial(layout_actions.show_hide_input_target_media_panel, self))
        self.faceMaskCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        self.faceCompareCheckBox.clicked.connect(partial(video_control_actions.process_compare_checkboxes, self))
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=COMMON_LAYOUT_DATA, layoutWidget=self.commonWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SWAPPER_LAYOUT_DATA, layoutWidget=self.swapWidgetsLayout, data_type='parameter')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=SETTINGS_LAYOUT_DATA, layoutWidget=self.settingsWidgetsLayout, data_type='control')
        layout_actions.add_widgets_to_tab_layout(self, LAYOUT_DATA=FACE_EDITOR_LAYOUT_DATA, layoutWidget=self.faceEditorWidgetsLayout, data_type='parameter')
        self.outputFolderButton.clicked.connect(partial(list_view_actions.select_output_media_folder, self))
        common_widget_actions.create_control(self, 'OutputMediaFolder', '')
        self.current_widget_parameters = ParametersDict(copy.deepcopy(self.default_parameters), self.default_parameters)
        video_control_actions.reset_media_buttons(self)
        self.clipboard_timer.timeout.connect(self.check_clipboard)
        self.cancelBatchButton.clicked.connect(self.cancel_batch_processing)
        self.batchProgressWidget.hide()
        font = self.vramProgressBar.font()
        font.setBold(True)
        self.vramProgressBar.setFont(font)
        common_widget_actions.update_gpu_memory_progressbar(self)
        self.tabWidget.setCurrentIndex(0)
        
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.initialize_variables()
        self.initialize_widgets()
        self.load_last_workspace()

    def open_embedding_editor(self):
        if self.embedding_editor_window is None:
            self.embedding_editor_window = EmbeddingGUI()
        self.embedding_editor_window.showMaximized()

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        if self.scene.items():
            pixmap_item = self.scene.items()[0]
            scene_rect = pixmap_item.boundingRect()
            self.graphicsViewFrame.setSceneRect(scene_rect)
            graphics_view_actions.fit_image_to_view(self, pixmap_item, scene_rect )

    def keyPressEvent(self, event):
        match event.key():
            case QtCore.Qt.Key_F11: video_control_actions.view_fullscreen(self)
            case QtCore.Qt.Key_V: video_control_actions.advance_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_C: video_control_actions.rewind_video_slider_by_n_frames(self, n=1)
            case QtCore.Qt.Key_D: video_control_actions.advance_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_A: video_control_actions.rewind_video_slider_by_n_frames(self, n=30)
            case QtCore.Qt.Key_Z: self.videoSeekSlider.setValue(0)
            case QtCore.Qt.Key_Space: self.buttonMediaPlay.click()
            case QtCore.Qt.Key_R: self.buttonMediaRecord.click()
            case QtCore.Qt.Key_F:
                if event.modifiers() & QtCore.Qt.KeyboardModifier.AltModifier: video_control_actions.remove_video_slider_marker(self)
                else: video_control_actions.add_video_slider_marker(self)
            case QtCore.Qt.Key_W: video_control_actions.move_slider_to_nearest_marker(self, 'next')
            case QtCore.Qt.Key_Q: video_control_actions.move_slider_to_nearest_marker(self, 'previous')
            case QtCore.Qt.Key_S: self.swapfacesButton.click()
    
    def toggle_clipboard_monitoring(self, checked):
        if checked:
            self.clipboard_timer.start(1000)
            print("Clipboard monitoring started.")
        else:
            self.clipboard_timer.stop()
            print("Clipboard monitoring stopped.")

    def check_clipboard(self):
        clipboard = QtGui.QGuiApplication.clipboard()
        mime_data = clipboard.mimeData()
        potential_files = []
        new_clipboard_content = ""

        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls:
                new_clipboard_content = urls[0].toString()
                for url in urls:
                    if url.isLocalFile(): potential_files.append(url.toLocalFile())
        elif mime_data.hasText():
            new_clipboard_content = mime_data.text()
            if new_clipboard_content: potential_files = new_clipboard_content.splitlines()

        if new_clipboard_content and new_clipboard_content != self.last_clipboard_content:
            self.last_clipboard_content = new_clipboard_content
            valid_files = []
            current_files_in_list = {btn.media_path for btn in self.target_videos.values()}
            for f_path in potential_files:
                f_path = f_path.strip().strip('"') 
                if os.path.isfile(f_path) and (misc_helpers.is_image_file(f_path) or misc_helpers.is_video_file(f_path)):
                     if f_path not in current_files_in_list and f_path not in self.clipboard_file_queue:
                        valid_files.append(f_path)
            if valid_files:
                if self.is_batch_processing_active or (self.video_loader_worker and self.video_loader_worker.isRunning()):
                    self.clipboard_file_queue.extend(valid_files)
                    print(f"Queued {len(valid_files)} new files from clipboard as a process is running.")
                else:
                    self.add_files_to_target_list(valid_files)

    def add_files_to_target_list(self, files_list):
        if self.control.get('AutoProcessTargetToggle', False):
            self.process_batch_on_load_finish = True
        if self.control.get('AutoBatchSaveFrameToggle', False):
            self.process_batch_save_frame_on_load_finish = True
            self.files_for_batch_save_frame.extend(files_list)

        if not self.video_loader_worker or not self.video_loader_worker.isRunning():
            self.video_loader_worker = ui_workers.TargetMediaLoaderWorker(main_window=self, files_list=files_list)
            self.video_loader_worker.thumbnail_ready.connect(partial(list_view_actions.add_media_thumbnail_to_target_videos_list, self))
            self.video_loader_worker.finished.connect(self.on_adding_files_finished)
            self.video_loader_worker.start()
        else:
            print("Warning: Media loader is already running. Could not add files from clipboard.")
            self.process_batch_on_load_finish = False
            self.process_batch_save_frame_on_load_finish = False
    
    def on_adding_files_finished(self):
        # This function is the entry point for auto-processing. It must be protected by the lock.
        if self.is_batch_processing_active:
            print("Auto-process trigger ignored: a batch process is already active.")
            return

        # Prioritize full batch processing over single-frame save
        if self.process_batch_on_load_finish:
            self.process_batch_on_load_finish = False
            print("Auto-processing triggered after file loading finished.")
            self.start_batch_processing()
        elif self.process_batch_save_frame_on_load_finish:
            self.process_batch_save_frame_on_load_finish = False
            files_to_process = getattr(self, 'files_for_batch_save_frame', [])
            if files_to_process:
                self.start_clipboard_batch_save_frame(files_to_process)
            self.files_for_batch_save_frame.clear()

    def start_batch_processing(self):
        if self.is_batch_processing_active:
            print("Batch processing request ignored: a process is already running.")
            return

        output_to_target = self.control.get('OutputToTargetLocationToggle', False)
        if not output_to_target:
            output_dir = self.outputFolderLineEdit.text()
            if not output_dir or not Path(output_dir).is_dir():
                common_widget_actions.create_and_show_messagebox(
                    self, "Output Directory Not Set", "Please set a valid output directory in Settings or enable 'Output to Target Location'.", self)
                return
        
        all_media_widgets = [self.targetVideosList.itemWidget(self.targetVideosList.item(i)) for i in range(self.targetVideosList.count()) if self.targetVideosList.item(i)]
        if not all_media_widgets:
            if not self.clipboard_file_queue:
                 common_widget_actions.create_and_show_messagebox(self, "No Items Found", "There are no items in the target list to process.", self)
            return
        
        # --- FIX: Set lock and show progress bar BEFORE starting initialization ---
        self.is_batch_processing_active = True
        self.batchProgressWidget.show()
        self.batchProgressLabel.setText("Initializing...")
        self.cancelBatchButton.setEnabled(True)
        self.initialize_and_start_batch(all_media_widgets)

    def initialize_and_start_batch(self, media_widgets_to_process):
        if media_widgets_to_process:
            self.check_init_status(media_widgets_to_process, 0)
        else:
            # Failsafe if called with empty list
            self.is_batch_processing_active = False
            self.batchProgressWidget.hide()

    def check_init_status(self, media_widgets_to_process, current_index):
        if current_index >= len(media_widgets_to_process):
            print("Could not find any faces in any of the provided media. Aborting batch process.")
            common_widget_actions.create_and_show_messagebox(
                self, "No Faces Found", "Could not detect any faces in any of the target items. Please provide at least one item with a clear face.", self)
            # --- FIX: Release lock and hide progress bar on failure ---
            self.is_batch_processing_active = False
            self.batchProgressWidget.hide()
            return

        widget_to_try = media_widgets_to_process[current_index]
        print(f"Attempting to initialize with: {os.path.basename(widget_to_try.media_path)}")
        widget_to_try.click()

        # Increased timer slightly for more robustness on slower systems
        QTimer.singleShot(200, lambda: self.verify_init(media_widgets_to_process, current_index))

    def verify_init(self, media_widgets_to_process, current_index):
        if self.target_faces:
            print("Initialization successful. Found faces in the selected media.")
            all_media_paths = [widget.media_path for widget in media_widgets_to_process]
            self.deferred_batch_start(all_media_paths)
        else:
            print(f"Initialization failed for media at index {current_index}. Trying next...")
            self.check_init_status(media_widgets_to_process, current_index + 1)

    def deferred_batch_start(self, media_files_to_process):
        # Lock is already set, so we can proceed safely
        print("Deferred batch start: All systems go.")
        self.items_in_current_batch = [item for item in self.targetVideosList.findItems("*", QtCore.Qt.MatchWildcard) if self.targetVideosList.itemWidget(item).media_path in media_files_to_process]

        self.batchProgressBar.setValue(0)
        self.batchProgressLabel.setText("Starting batch process...")
        
        self.batch_worker = ui_workers.BatchProcessorWorker(self, media_files_to_process)
        self.batch_worker.progress.connect(self.update_batch_progress)
        self.batch_worker.output_directory_updated.connect(self.update_last_output_directory)
        self.batch_worker.finished.connect(self.on_batch_processing_finished)
        self.batch_worker.start()

    def start_clipboard_batch_save_frame(self, media_files):
        if self.is_batch_processing_active:
            print("Process ignored: another batch process is already running.")
            return

        # --- FIX: Set lock and show progress bar ---
        self.is_batch_processing_active = True
        print("Starting clipboard-triggered batch save frame.")
        self.batchProgressBar.setValue(0)
        self.batchProgressLabel.setText("Starting batch save frame...")
        self.batchProgressWidget.show()
        self.cancelBatchButton.setEnabled(True)

        self.clipboard_batch_save_worker = ui_workers.ClipboardBatchSaveFrameWorker(self, media_files)
        self.clipboard_batch_save_worker.progress.connect(self.update_batch_progress)
        self.clipboard_batch_save_worker.output_directory_updated.connect(self.update_last_output_directory)
        self.clipboard_batch_save_worker.finished.connect(self.on_batch_processing_finished)
        self.clipboard_batch_save_worker.start()
        
    def update_batch_progress(self, file_idx, total_files, filename, frame_idx, total_frames):
        self.batchProgressBar.setMaximum(total_files)
        self.batchProgressBar.setValue(file_idx)

        if total_frames > 0:
            progress_percent = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
            self.batchProgressLabel.setText(f"Processing: {filename} ({file_idx+1}/{total_files}) - Frame {frame_idx}/{total_frames} ({progress_percent:.0f}%)")
        else:
            self.batchProgressLabel.setText(f"Processing: {filename} ({file_idx+1}/{total_files})")
    
    @QtCore.Slot(str)
    def update_last_output_directory(self, path):
        self.last_output_directory = path

    def cancel_batch_processing(self):
        worker_cancelled = False
        if hasattr(self, 'batch_worker') and self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.stop()
            worker_cancelled = True
        if hasattr(self, 'clipboard_batch_save_worker') and self.clipboard_batch_save_worker and self.clipboard_batch_save_worker.isRunning():
            self.clipboard_batch_save_worker.stop()
            worker_cancelled = True
        if hasattr(self, 'frame_emb_batch_worker') and self.frame_emb_batch_worker and self.frame_emb_batch_worker.isRunning():
            self.frame_emb_batch_worker.stop()
            worker_cancelled = True
        
        if worker_cancelled:
            self.batchProgressLabel.setText("Cancelling...")
            self.cancelBatchButton.setEnabled(False)

    def on_batch_processing_finished(self, message):
        self.batchProgressBar.setValue(self.batchProgressBar.maximum())
        self.batchProgressLabel.setText(message)
        self.cancelBatchButton.setEnabled(False)
        QtCore.QTimer.singleShot(2000, self.cleanup_after_batch)
        
    def cleanup_after_batch(self):
        self.batchProgressWidget.hide()
        self.cancelBatchButton.setEnabled(True)

        if self.control.get('ClearTargetAfterProcessingToggle', False):
            print(f"Clearing {len(self.items_in_current_batch)} processed files from the target list.")
            for item in self.items_in_current_batch:
                try:
                    widget = self.targetVideosList.itemWidget(item)
                    if widget:
                        if self.selected_video_button == widget: self.selected_video_button = None
                        if widget.media_id in self.target_videos: self.target_videos.pop(widget.media_id)
                        row = self.targetVideosList.row(item)
                        if row != -1: self.targetVideosList.takeItem(row)
                except Exception as e:
                    print(f"Error removing item from list: {e}")
            self.items_in_current_batch.clear()
            if self.targetVideosList.count() == 0:
                list_view_actions.clear_all_target_media(self)
        
        self.batch_worker = None
        self.frame_emb_batch_worker = None
        self.clipboard_batch_save_worker = None

        # --- FIX: Release the lock at the very end of cleanup ---
        self.is_batch_processing_active = False

        # After releasing the lock, check for and process any queued files
        if self.clipboard_file_queue:
            print(f"Processing {len(self.clipboard_file_queue)} queued files.")
            files_to_add = self.clipboard_file_queue.copy()
            self.clipboard_file_queue.clear()
            self.add_files_to_target_list(files_to_add)
        
    def closeEvent(self, event):
        print("MainWindow: closeEvent called.")
        self.video_processor.stop_processing()
        list_view_actions.clear_stop_loading_input_media(self)
        list_view_actions.clear_stop_loading_target_media(self)
        save_load_actions.save_current_workspace(self, 'last_workspace.json')
        event.accept()

    def load_last_workspace(self):
        if Path('last_workspace.json').is_file():
            load_dialog = widget_components.LoadLastWorkspaceDialog(self)
            load_dialog.exec_()

    def save_last_workspace(self):
        pass

    def open_last_output_directory(self):
        path_to_open = ""
        if self.control.get('OutputToTargetLocationToggle', False):
            path_to_open = self.last_output_directory
            if not path_to_open:
                common_widget_actions.create_and_show_messagebox(
                    self, "No Processed Files", "No files have been processed with 'Output to Target Location' enabled yet.", self)
                return
        else:
            path_to_open = self.outputFolderLineEdit.text()

        if not path_to_open or not os.path.isdir(path_to_open):
            common_widget_actions.create_and_show_messagebox(
                self, "Invalid Directory", f"The specified output directory does not exist:\n{path_to_open}", self)
            return

        QDesktopServices.openUrl(QUrl.fromLocalFile(path_to_open))

    def toggle_embedding_sort(self):
        modes = ["Off", "A-Z", "Z-A"]
        if self.embedding_sort_mode == "Off":
            self.unsorted_embedding_ids = list(self.merged_embeddings.keys())
        try:
            current_index = modes.index(self.embedding_sort_mode)
        except ValueError:
            current_index = -1
        next_index = (current_index + 1) % len(modes)
        self.embedding_sort_mode = modes[next_index]
        self.sortEmbeddingsButton.setText(f"Sorting: {self.embedding_sort_mode}")
        save_load_actions.reorder_embeddings_from_data(self, self.embedding_sort_mode)