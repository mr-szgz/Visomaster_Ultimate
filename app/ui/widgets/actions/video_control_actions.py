from typing import TYPE_CHECKING
import copy
import os
from functools import partial

import cv2
import numpy
from PIL import Image
from PySide6 import QtGui,QtWidgets,QtCore

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow
import app.helpers.miscellaneous as misc_helpers
from app.ui.widgets.actions import common_actions as common_widget_actions
from app.ui.widgets.actions import graphics_view_actions
import app.ui.widgets.actions.layout_actions as layout_actions

def set_up_video_seek_line_edit(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    videoSeekLineEdit = main_window.videoSeekLineEdit
    videoSeekLineEdit.setAlignment(QtCore.Qt.AlignCenter)
    videoSeekLineEdit.setText('0')
    videoSeekLineEdit.setValidator(QtGui.QIntValidator(0, video_processor.max_frame_number))  # Restrict input to numbers 

def set_up_video_seek_slider(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()  # Store unique tick positions
    main_window.videoSeekSlider.setTickPosition(QtWidgets.QSlider.TickPosition.TicksBelow)  # Default position for tick marks

    def add_marker_and_paint(self: QtWidgets.QSlider, value=None):
        """Add a tick mark at a specific slider value."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if self.minimum() <= value <= self.maximum() and value not in self.markers:
            self.markers.add(value)
            self.update()

    def remove_marker_and_paint(self:QtWidgets.QSlider, value=None):
        """Remove a tick mark."""
        if value is None or isinstance(value, bool):  # Default to current slider value
            value = self.value()
        if value in self.markers:
            self.markers.remove(value)
            self.update()

    def paintEvent(self: QtWidgets.QSlider, event:QtGui.QPaintEvent):
        # Dont need a seek slider if the current selected file is an image
        if main_window.video_processor.file_type=='image':
            return super(QtWidgets.QSlider, self).paintEvent(event)
        # Set up the painter and style option
        painter = QtWidgets.QStylePainter(self)
        opt = QtWidgets.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()

        # Get groove and handle geometry
        groove_rect = style.subControlRect(
            QtWidgets.QStyle.ComplexControl.CC_Slider, opt, QtWidgets.QStyle.SubControl.SC_SliderGroove
        )
        groove_y = (groove_rect.top() + groove_rect.bottom()) // 2  # Groove's vertical center
        groove_start = groove_rect.left()
        groove_end = groove_rect.right()
        groove_width = groove_end - groove_start

        # Calculate handle position based on the current slider value
        normalized_value = (self.value() - self.minimum()) / (self.maximum() - self.minimum())
        handle_center_x = groove_start + normalized_value * groove_width

        # Make the handle thinner
        handle_width = 5  # Fixed width for thin handle
        handle_height = groove_rect.height()  # Slightly shorter than groove height
        handle_left_x = handle_center_x - (handle_width // 2)
        handle_top_y = groove_y - (handle_height // 2)

        # Define the handle rectangle
        handle_rect = QtCore.QRect(
            handle_left_x, handle_top_y, handle_width, handle_height
        )

        # Draw the groove
        painter.setPen(QtGui.QPen(QtGui.QColor("gray"), 3))  # Groove color and thickness
        painter.drawLine(groove_start, groove_y, groove_end, groove_y)

        # Draw the thin handle
        painter.setPen(QtGui.QPen(QtGui.QColor("white"), 1))  # Handle border color
        painter.setBrush(QtGui.QBrush(QtGui.QColor("white")))  # Handle fill color
        painter.drawRect(handle_rect)

        # Draw markers (if any)
        if self.markers:
            painter.setPen(QtGui.QPen(QtGui.QColor("#e8483c"), 2))  # Marker color and thickness
            for value in sorted(self.markers):
                # Calculate marker position
                marker_normalized_value = (value - self.minimum()) / (self.maximum() - self.minimum())
                marker_x = groove_start + marker_normalized_value * groove_width
                painter.drawLine(marker_x, groove_rect.top(), marker_x, groove_rect.bottom())

    main_window.videoSeekSlider.add_marker_and_paint = partial(add_marker_and_paint, main_window.videoSeekSlider)
    main_window.videoSeekSlider.remove_marker_and_paint = partial(remove_marker_and_paint, main_window.videoSeekSlider)
    main_window.videoSeekSlider.paintEvent = partial(paintEvent, main_window.videoSeekSlider)

def add_video_slider_marker(main_window: 'MainWindow'):
    if main_window.selected_video_button.file_type!='video':
        common_widget_actions.create_and_show_messagebox(main_window, 'Markers Not Available', 'Markers can only be used for videos!', main_window.videoSeekSlider)
        return
    current_position = int(main_window.videoSeekSlider.value())
    # print("current_position", current_position)
    if not main_window.target_faces:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Target Face Found', 'You need to have atleast one target face to create a marker', main_window.videoSeekSlider)
    elif main_window.markers.get(current_position):
        common_widget_actions.create_and_show_messagebox(main_window, 'Marker Already Exists!', 'A Marker already exists for this position!', main_window.videoSeekSlider)
    else:
        add_marker(main_window, copy.deepcopy(main_window.parameters), main_window.control.copy(), current_position)

def remove_video_slider_marker(main_window: 'MainWindow'):
    if main_window.selected_video_button.file_type!='video':
        common_widget_actions.create_and_show_messagebox(main_window, 'Markers Not Available', 'Markers can only be used for videos!', main_window.videoSeekSlider)
        return
    current_position = int(main_window.videoSeekSlider.value())
    # print("current_position", current_position)
    if main_window.markers.get(current_position):
        remove_marker(main_window, current_position)
    else:
        common_widget_actions.create_and_show_messagebox(main_window, 'No Marker Found!', 'No Marker Found for this position!', main_window.videoSeekSlider)

def add_marker(main_window: 'MainWindow', parameters, control, position,):
    main_window.videoSeekSlider.add_marker_and_paint(position)
    main_window.markers[position] = {'parameters': parameters, 'control': control}
    print(f"Marker Added for Frame: {position}")

def remove_marker(main_window: 'MainWindow', position):
    if main_window.markers.get(position):
        main_window.videoSeekSlider.remove_marker_and_paint(position)
        main_window.markers.pop(position)
        print(f"Marker Removed from position: {position}")

def remove_all_markers(main_window: 'MainWindow'):
    for marker_position in list(main_window.markers.keys()):
        remove_marker(main_window, marker_position)

def move_slider_to_nearest_marker(main_window: 'MainWindow', direction: str):
    """
    Move the slider to the nearest marker in the specified direction.

    :param direction: 'next' to move to the next marker, 'previous' to move to the previous marker.
    """
    new_position = None
    current_position = int(main_window.videoSeekSlider.value())
    markers = sorted(main_window.markers.keys())
    if direction == "next":
        filtered_markers = [marker for marker in markers if marker > current_position]
        new_position = filtered_markers[0] if filtered_markers else None
    elif direction == "previous":
        filtered_markers = [marker for marker in markers if marker < current_position]
        new_position = filtered_markers[-1] if filtered_markers else None

    if new_position is not None:
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

# Wrappers for specific directions
def move_slider_to_next_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "next")

def move_slider_to_previous_nearest_marker(main_window: 'MainWindow'):
    move_slider_to_nearest_marker(main_window, "previous")

def remove_face_parameters_and_control_from_markers(main_window: 'MainWindow', face_id):
    for _, marker_data in main_window.markers.items():
        marker_data['parameters'].pop(face_id)
        # If the parameters is empty, then there is no longer any marker to be set for any target face
        if not marker_data['parameters']:
            delete_all_markers(main_window)
            break

def advance_video_slider_by_n_frames(main_window: 'MainWindow', n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position + n
        if new_position > video_processor.max_frame_number:
            new_position = video_processor.max_frame_number
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

def rewind_video_slider_by_n_frames(main_window: 'MainWindow', n=30):
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        current_position = int(main_window.videoSeekSlider.value())
        new_position = current_position - n
        if new_position < 0:
            new_position = 0
        main_window.videoSeekSlider.setValue(new_position)
        main_window.video_processor.process_current_frame()

def delete_all_markers(main_window: 'MainWindow'):
    main_window.videoSeekSlider.markers = set()
    main_window.videoSeekSlider.update()
    main_window.markers = {}

def view_fullscreen(main_window: 'MainWindow'):

    if main_window.is_full_screen:
        main_window.showNormal()  # Exit full-screen mode
        main_window.menuBar().show()
    else:
        main_window.showFullScreen()  # Enter full-screen mode
        main_window.menuBar().hide()

    main_window.is_full_screen = not main_window.is_full_screen

def enable_zoom_and_pan(view: QtWidgets.QGraphicsView):
    SCALE_FACTOR = 1.1
    view.zoom_value = 0  # Track zoom level
    view.last_scale_factor = 1.0  # Track the last scale factor (1.0 = no scaling)
    view.is_panning = False  # Track whether panning is active
    view.pan_start_pos = QtCore.QPoint()  # Store the initial mouse position for panning

    def zoom(self:QtWidgets.QGraphicsView, step=False):
        """Zoom in or out by a step."""
        if not step:
            factor = self.last_scale_factor
        else:
            self.zoom_value += step
            factor = SCALE_FACTOR ** step
            self.last_scale_factor *= factor  # Update the last scale factor
        if factor > 0:
            self.scale(factor, factor)

    def wheelEvent(self:QtWidgets.QGraphicsView, event:QtGui.QWheelEvent):
        """Handle mouse wheel event for zooming."""
        delta = event.angleDelta().y()
        if delta != 0:
            zoom(self, delta // abs(delta))
    
    def reset_zoom(self:QtWidgets.QGraphicsView):
        # print("Called reset_zoom()")
        # Reset zoom level to fit the view.
        self.zoom_value = 0
        if not self.scene():
            return
        items = self.scene().items()
        if not items:
            return
        rect = self.scene().itemsBoundingRect()
        self.setSceneRect(rect)
        unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
        self.scale(1 / unity.width(), 1 / unity.height())
        view_rect = self.viewport().rect()
        scene_rect = self.transform().mapRect(rect)
        factor = min(view_rect.width() / scene_rect.width(),
                    view_rect.height() / scene_rect.height())
        self.scale(factor, factor)

    def mousePressEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse press event for panning."""
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.is_panning = True
            self.pan_start_pos = event.pos()  # Store the initial mouse position
            self.setCursor(QtCore.Qt.ClosedHandCursor)  # Change cursor to indicate panning
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mousePressEvent(self, event)

    def mouseMoveEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse move event for panning."""
        if self.is_panning:
            # Calculate the distance moved
            delta = event.pos() - self.pan_start_pos
            self.pan_start_pos = event.pos()  # Update the start position
            # Translate the view
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mouseMoveEvent(self, event)

    def mouseReleaseEvent(self: QtWidgets.QGraphicsView, event: QtGui.QMouseEvent):
        """Handle mouse release event for panning."""
        if event.button() == QtCore.Qt.MouseButton.RightButton:
            self.is_panning = False
            self.setCursor(QtCore.Qt.ArrowCursor)  # Reset the cursor
        else:
            # Explicitly call the base class implementation
            QtWidgets.QGraphicsView.mouseReleaseEvent(self, event)

    # Attach methods to the view
    view.zoom = partial(zoom, view)
    view.reset_zoom = partial(reset_zoom, view)
    view.wheelEvent = partial(wheelEvent, view)
    view.mousePressEvent = partial(mousePressEvent, view)
    view.mouseMoveEvent = partial(mouseMoveEvent, view)
    view.mouseReleaseEvent = partial(mouseReleaseEvent, view)

    # view.zoom = zoom.__get__(view)
    # view.reset_zoom = reset_zoom.__get__(view)
    # view.wheelEvent = wheelEvent.__get__(view)

    # Set anchors for better interaction
    view.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)
    view.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorUnderMouse)


def play_video(main_window: 'MainWindow', checked: bool):
    video_processor = main_window.video_processor
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("play_video: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
            return
        print("play_video: Starting video processing.")
        set_play_button_icon_to_stop(main_window)
        video_processor.process_video()
    else:
        video_processor = main_window.video_processor
        # print("play_video: Stopping video processing.")
        set_play_button_icon_to_play(main_window)
        video_processor.stop_processing()
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(False)
        main_window.buttonMediaRecord.blockSignals(False)
        set_record_button_icon_to_play(main_window)


def record_video(main_window: 'MainWindow', checked: bool):
    video_processor = main_window.video_processor
    # Dont record webcam capture
    if video_processor.file_type == 'webcam':
        main_window.buttonMediaRecord.blockSignals(True)
        main_window.buttonMediaRecord.setChecked(not checked)
        main_window.buttonMediaRecord.blockSignals(False)
        return
    
    if checked:
        if video_processor.processing or video_processor.current_frame_number==video_processor.max_frame_number:
            print("record_video: Video already playing. Stopping the current video before starting a new one.")
            video_processor.stop_processing()
            return
        if not main_window.control.get('OutputMediaFolder','').strip():
            common_widget_actions.create_and_show_messagebox(main_window, 'No Output Folder Selected','Please select an Output folder to save the Videos before recording!', main_window)
            main_window.buttonMediaRecord.setChecked(False)
            return
        if not misc_helpers.is_ffmpeg_in_path():
            common_widget_actions.create_and_show_messagebox(main_window, 'FFMPEG Not Found','FFMPEG was not found in your system. Check your installation!', main_window)
            main_window.buttonMediaRecord.setChecked(False)
            return
        video_processor.recording = True
        main_window.buttonMediaPlay.setChecked(True)
        set_record_button_icon_to_stop(main_window)

    else:
        main_window.buttonMediaPlay.setChecked(False)
        video_processor.stop_processing()
        set_play_button_icon_to_play(main_window)
        set_record_button_icon_to_play(main_window)

def set_record_button_icon_to_play(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
    main_window.buttonMediaRecord.setToolTip("Start Recording")
def set_record_button_icon_to_stop(main_window: 'MainWindow'):
    main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
    main_window.buttonMediaRecord.setToolTip("Stop Recording")

def set_play_button_icon_to_play(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
    main_window.buttonMediaPlay.setToolTip("Play")

def set_play_button_icon_to_stop(main_window: 'MainWindow'):
    main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
    main_window.buttonMediaPlay.setToolTip("Stop")

def reset_media_buttons(main_window: 'MainWindow'):
    # Rest the state and icons of the buttons without triggering Onchange methods
    main_window.buttonMediaPlay.blockSignals(True)
    main_window.buttonMediaPlay.setChecked(False)
    main_window.buttonMediaPlay.blockSignals(False)
    main_window.buttonMediaRecord.blockSignals(True)
    main_window.buttonMediaRecord.setChecked(False)
    main_window.buttonMediaRecord.blockSignals(False)
    set_play_button_icon(main_window)
    set_record_button_icon(main_window)


def set_play_button_icon(main_window: 'MainWindow'):
    if main_window.buttonMediaPlay.isChecked(): 
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_on.png"))
        main_window.buttonMediaPlay.setToolTip("Stop")
    else:
        main_window.buttonMediaPlay.setIcon(QtGui.QIcon(":/media/media/play_off.png"))
        main_window.buttonMediaPlay.setToolTip("Play")

def set_record_button_icon(main_window: 'MainWindow'):
    if main_window.buttonMediaRecord.isChecked(): 
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_on.png"))
        main_window.buttonMediaRecord.setToolTip("Stop Recording")
    else:
        main_window.buttonMediaRecord.setIcon(QtGui.QIcon(":/media/media/rec_off.png"))
        main_window.buttonMediaRecord.setToolTip("Start Recording")

# @misc_helpers.benchmark
@QtCore.Slot(int)
def on_change_video_seek_slider(main_window: 'MainWindow', new_position=0):
    # print("Called on_change_video_seek_slider()")
    video_processor = main_window.video_processor

    was_processing = video_processor.stop_processing()
    if was_processing:
        print("on_change_video_seek_slider: Processing in progress. Stopping current processing.")

    video_processor.current_frame_number = new_position
    video_processor.next_frame_to_display = new_position
    if video_processor.media_capture:
        video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
        ret, frame = misc_helpers.read_frame(video_processor.media_capture)
        if ret:
            pixmap = common_widget_actions.get_pixmap_from_frame(main_window, frame)
            graphics_view_actions.update_graphics_view(main_window, pixmap, new_position)
            if video_processor.current_frame_number == video_processor.max_frame_number:
                video_processor.media_capture.set(cv2.CAP_PROP_POS_FRAMES, new_position)
            update_parameters_and_control_from_marker(main_window, new_position)
            update_widget_values_from_markers(main_window, new_position)

    # Do not automatically restart the video, let the user press Play to resume
    # print("on_change_video_seek_slider: Video stopped after slider movement.")

def update_parameters_and_control_from_marker(main_window: 'MainWindow', new_position: int):
    if main_window.markers.get(new_position):
        main_window.parameters = copy.deepcopy(main_window.markers[new_position]['parameters'])
        main_window.control.update(main_window.markers[new_position]['control'].copy())

def update_widget_values_from_markers(main_window: 'MainWindow', new_position: int):
    if main_window.markers.get(new_position):
        if main_window.selected_target_face_id is not None:
            common_widget_actions.set_widgets_values_using_face_id_parameters(main_window, main_window.selected_target_face_id)
            common_widget_actions.set_control_widgets_values(main_window, enable_exec_func=False)

def on_slider_moved(main_window: 'MainWindow'):
    # print("Called on_slider_moved()")
    position = main_window.videoSeekSlider.value()
    # print(f"\nSlider Moved. position: {position}\n")

def on_slider_pressed(main_window: 'MainWindow'):

    position = main_window.videoSeekSlider.value()
    # print(f"\nSlider Pressed. position: {position}\n")

# @misc_helpers.benchmark
def on_slider_released(main_window: 'MainWindow'):
    # print("Called on_slider_released()")

    new_position = main_window.videoSeekSlider.value()
    # print(f"\nSlider released. New position: {new_position}\n")
    # Perform the update to the new frame
    video_processor = main_window.video_processor
    if video_processor.media_capture:
        video_processor.process_current_frame()  # Process the current frame

def process_swap_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_edit_faces(main_window: 'MainWindow'):
    video_processor = main_window.video_processor
    video_processor.process_current_frame()

def process_compare_checkboxes(main_window: 'MainWindow'):
    main_window.video_processor.process_current_frame()
    layout_actions.fit_image_to_view_onchange(main_window)

def save_current_frame_to_file(main_window: 'MainWindow'):
    """
    存『目前畫面』為一張圖片（JPEG）。
    會自動依目前使用的來源 name（embedding 或 input face）在輸出資料夾下建立子資料夾。
    """
    def _sanitize(name: str) -> str:
        return "".join([c for c in name if c.isalnum() or c in "._-"]).rstrip()

    def _get_active_source_name() -> str | None:
        """
        來源優先序：
        1) 目前 target face 所指派的 merged embeddings（若>1，取第一個）
        2) 目前 target face 所指派的 input faces（若>1，取第一個；用檔名 stem 當 name）
        """
        tf_btn = main_window.cur_selected_target_face_button
        if not tf_btn:
            return None

        # 1) merged embeddings
        if getattr(tf_btn, "assigned_merged_embeddings", None):
            # 取第一個 id
            emb_id = next(iter(tf_btn.assigned_merged_embeddings.keys()), None)
            if emb_id and emb_id in main_window.merged_embeddings:
                emb_btn = main_window.merged_embeddings[emb_id]
                name = getattr(emb_btn, "embedding_name", None)
                if name:
                    return _sanitize(name)

        # 2) input faces
        if getattr(tf_btn, "assigned_input_faces", None):
            face_id = next(iter(tf_btn.assigned_input_faces.keys()), None)
            if face_id and face_id in main_window.input_faces:
                face_btn = main_window.input_faces[face_id]
                base = os.path.splitext(os.path.basename(face_btn.media_path))[0]
                return _sanitize(base)

        return None

    # 先決定「基底輸出資料夾」
    output_to_target = main_window.control.get('OutputToTargetLocationToggle', False)
    if output_to_target:
        if main_window.selected_video_button and main_window.selected_video_button.media_path:
            base_output_folder = os.path.dirname(main_window.selected_video_button.media_path)
        else:
            common_widget_actions.create_and_show_messagebox(
                main_window, 'No target media selected',
                'No target media selected to determine output path.', main_window
            )
            return
    else:
        base_output_folder = main_window.outputFolderLineEdit.text()

    if not base_output_folder:
        common_widget_actions.create_and_show_messagebox(
            main_window, 'No Output Folder Selected',
            'Please select an Output folder in Settings, or enable "Output to Target Location".', main_window
        )
        return

    # 取得目前 frame
    frame = main_window.video_processor.current_frame.copy()
    if not isinstance(frame, numpy.ndarray):
        common_widget_actions.create_and_show_messagebox(
            main_window, 'No Frame Available', 'Cannot access the current frame!',
            parent_widget=main_window.saveImageButton
        )
        return

    # 依來源 name 建立子資料夾
    source_name = _get_active_source_name()
    output_folder = base_output_folder
    cluster_by_source = main_window.control.get('ClusterOutputBySourceToggle', True)
    if cluster_by_source and source_name:
        output_folder = os.path.join(base_output_folder, source_name)

    try:
        os.makedirs(output_folder, exist_ok=True)
    except Exception as e:
        common_widget_actions.create_and_show_messagebox(
            main_window, 'Create Folder Failed',
            f'Failed to create folder:\n{output_folder}\n{e}',
            parent_widget=main_window.saveImageButton
        )
        return

    # 用 helper 先拿到原本規則的「基礎檔名」，再加前綴時間＋結尾亂碼，避免重名覆蓋
    tmp = misc_helpers.get_output_file_path(
        main_window.video_processor.media_path, output_folder, media_type='image'
    )
    base_noext, _ = os.path.splitext(tmp)
    stem = os.path.basename(base_noext)      # 原本檔名主體
    if cluster_by_source and source_name:
        stem = f"{stem}_{source_name}"

    dir_ = os.path.dirname(base_noext)
    import datetime, secrets, string
    ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    rand = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))

    save_filename = os.path.join(dir_, f"{ts}_{stem}_{rand}.jpg")

    # Debug
    print(f"[save_image] base_output_folder={base_output_folder}")
    print(f"[save_image] source_name={source_name}")
    print(f"[save_image] output_folder={output_folder}")
    print(f"[save_image] save_filename={save_filename}")

    try:
        # BGR -> RGB
        pil_image = Image.fromarray(frame[..., ::-1])
        pil_image.save(save_filename, 'JPEG', quality=85)
        common_widget_actions.create_and_show_toast_message(
            main_window, 'Image Saved', f'Saved Current Image to file: {save_filename}'
        )
    except Exception as e:
        common_widget_actions.create_and_show_messagebox(
            main_window, 'Error Saving Image',
            f'An error occurred while saving the image:\n{e}',
            parent_widget=main_window.saveImageButton
        )


class CurrentFrameEmbBatchWorker(QtCore.QThread):
    """
    只針對「目前 UI 顯示的那一偵」逐一切換所有 EMB 並各自存圖的背景執行緒。
    會沿用單張儲存的輸出邏輯（Output to Target / Output folder、子資料夾為 EMB 名稱、檔名加後綴）。
    透過 progress 訊號更新既有的 batchProgressBar，不會卡 UI。
    """
    progress = QtCore.Signal(int, int, str, int, int)  # file_idx, total_files, filename(label), frame_idx, total_frames
    finished = QtCore.Signal(str)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._is_running = True

    def stop(self):
        self._is_running = False

    def run(self):
        try:
            # ---- 本地 import，避免循環匯入 ----
            from PIL import Image
            import os, time
            try:
                # 若專案有 helper，就沿用檔名規則；沒有就走 fallback
                from app.ui.widgets.actions import misc_helpers
            except Exception:
                misc_helpers = None

            # ---- 取主要參照 ----
            tf_btn = getattr(self.main_window, 'cur_selected_target_face_button', None)
            merged_embeddings = getattr(self.main_window, 'merged_embeddings', {})
            video_processor = self.main_window.video_processor

            if not tf_btn or not merged_embeddings:
                self.finished.emit("No target face or embeddings to process.")
                return

            # ---- 產生「畫面指紋」與「等待畫面更新」的內部工具 ----
            def _frame_fingerprint(frame):
                try:
                    import numpy as np
                    if frame is None:
                        return None
                    # 輕量取樣指紋：每 32 像素取一點，只取單通道
                    sample = frame[::32, ::32, 0]
                    return int(sample.sum())
                except Exception:
                    return None

            def _wait_until_frame_updated(before_sig, timeout_ms=8000):
                """等待 current_frame 與 before_sig 不同（代表新 EMB 的結果已完成）"""
                t0 = time.time()

                # 若 video_processor 有 is_processing 旗標，先等它變 False
                if hasattr(video_processor, 'is_processing'):
                    while self._is_running and getattr(video_processor, 'is_processing', False):
                        QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
                        self.msleep(30)
                        if (time.time() - t0) * 1000 > timeout_ms:
                            return False

                # 再用指紋判斷實際畫面是否更新
                while self._is_running and (time.time() - t0) * 1000 <= timeout_ms:
                    cur = getattr(video_processor, 'current_frame', None)
                    sig = _frame_fingerprint(cur)
                    if sig is not None and sig != before_sig:
                        return True
                    QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
                    self.msleep(30)
                return False

            # ---- 取得輸出根目錄（沿用單張儲存規則）----
            output_to_target = self.main_window.control.get('OutputToTargetLocationToggle', False)
            if output_to_target:
                if self.main_window.selected_video_button and self.main_window.selected_video_button.media_path:
                    base_output_folder = os.path.dirname(self.main_window.selected_video_button.media_path)
                else:
                    self.finished.emit("No target media selected to determine output path.")
                    return
            else:
                base_output_folder = self.main_window.outputFolderLineEdit.text()
                if not base_output_folder:
                    self.finished.emit("No Output folder set.")
                    return

            # ---- 備份原本指派（結束時會復原）----
            original_assigned = dict(getattr(tf_btn, 'assigned_merged_embeddings', {}))

            total = len(merged_embeddings)
            for idx, (emb_id, emb_btn) in enumerate(merged_embeddings.items()):
                if not self._is_running:
                    break

                # 記住切 EMB 前的畫面指紋（避免上一張殘影）
                before_sig = _frame_fingerprint(getattr(video_processor, 'current_frame', None))

                # 1) 指派單一 EMB 到 target face
                tf_btn.assigned_merged_embeddings = {emb_id: emb_btn.embedding_store}
                tf_btn.calculate_assigned_input_embedding()

                # 2) 只處理「目前這一偵」
                video_processor.process_current_frame()

                # 3) 等待畫面真的更新（關鍵：避免 off-by-one 錯位）
                _wait_until_frame_updated(before_sig, timeout_ms=8000)

                frame = getattr(video_processor, 'current_frame', None)
                if frame is None:
                    self.progress.emit(idx + 1, total, "no_frame", 1, 1)
                    continue

                # 4) 子資料夾 = EMB 名稱
                cluster_by_source = self.main_window.control.get('ClusterOutputBySourceToggle', True)
                emb_name = getattr(emb_btn, 'embedding_name', str(emb_id))
                source_name = "".join([c for c in emb_name if c.isalnum() or c in "._-"]).rstrip() or "exported"
                output_folder = base_output_folder
                if cluster_by_source:
                    output_folder = os.path.join(base_output_folder, source_name)
                try:
                    os.makedirs(output_folder, exist_ok=True)
                except Exception:
                    self.progress.emit(idx + 1, total, source_name, 1, 1)
                    continue

                # 5) 檔名：保留原規則骨架，再加時間前綴＋6位亂碼，避免覆蓋
                if misc_helpers is not None:
                    tmp = misc_helpers.get_output_file_path(
                        video_processor.media_path, output_folder, media_type='image'
                    )
                    base_noext, _ = os.path.splitext(tmp)
                    stem = os.path.basename(base_noext)   # 原本檔名主體
                else:
                    base_name = os.path.splitext(os.path.basename(getattr(video_processor, 'media_path', 'frame')))[0]
                    stem = base_name

                if cluster_by_source and source_name:
                    stem = f"{stem}_{source_name}"

                import datetime, secrets, string
                ts = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
                rand = ''.join(secrets.choice(string.ascii_letters + string.digits) for _ in range(6))

                save_path = os.path.join(output_folder, f"{ts}_{stem}_{rand}.jpg")


                # 6) 寫檔（BGR -> RGB），不跳通知
                try:
                    pil = Image.fromarray(frame[..., ::-1])
                    pil.save(save_path, 'JPEG', quality=85)
                except Exception:
                    pass

                # 7) 更新進度列（沿用你的 update_batch_progress）
                self.progress.emit(idx + 1, total, source_name, 1, 1)

            # ---- 復原原本指派與畫面 ----
            tf_btn.assigned_merged_embeddings = original_assigned
            tf_btn.calculate_assigned_input_embedding()
            video_processor.process_current_frame()

            self.finished.emit("Embedding batch (current frame) completed." if self._is_running else "Embedding batch cancelled.")
        except Exception as e:
            self.finished.emit(f"Error: {e}")


    def _frame_fingerprint(self, frame):
        """
        產生一個很快的「畫面指紋」用於判斷畫面是否真的更新。
        取樣縮小後加總，避免重成本計算。
        """
        try:
            import numpy as np
            if frame is None:
                return None
            # 取樣：每 32 像素取一點，只拿單一通道避免計算太重
            sample = frame[::32, ::32, 0]
            return int(sample.sum())
        except Exception:
            return None

    def _wait_until_frame_updated(self, before_sig, timeout_ms=8000):
        """
        等待 current_frame 的指紋與 before_sig 不同（代表畫面已換成新 EMB 的結果）。
        期間持續讓事件循環跑，避免 UI 卡住；同時支援被 stop() 中斷。
        """
        import time
        t0 = time.time()

        # 如果你的 video_processor 有 is_processing 旗標，先優先等它變 False
        vp = self.main_window.video_processor
        if hasattr(vp, 'is_processing'):
            while self._is_running and getattr(vp, 'is_processing', False):
                QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
                self.msleep(30)
                if (time.time() - t0) * 1000 > timeout_ms:
                    return False

        # 再用指紋檢查畫面是否真的更新
        while self._is_running and (time.time() - t0) * 1000 <= timeout_ms:
            cur = getattr(self.main_window.video_processor, 'current_frame', None)
            sig = self._frame_fingerprint(cur)
            if sig is not None and sig != before_sig:
                return True
            QtCore.QCoreApplication.processEvents(QtCore.QEventLoop.AllEvents, 50)
            self.msleep(30)
        return False
        





def batch_save_current_frame_all_embeddings(main_window: 'MainWindow'):
    """
    啟動背景執行緒：僅針對目前 UI 這一偵，逐一切換所有 EMB，各自存檔。
    會顯示你現有的 batchProgressWidget，不會卡住 UI，也不會跳出通知。
    """
    # 基本檢查留在 UI 執行緒（必要時用既有 messagebox 提示）
    tf_btn = getattr(main_window, 'cur_selected_target_face_button', None)
    if not tf_btn:
        common_widget_actions.create_and_show_messagebox(
            main_window, 'No Target Face Selected',
            'Please select a target face before batch saving.',
            parent_widget=getattr(main_window, 'saveImageButton', None)
        )
        return

    merged_embeddings = getattr(main_window, 'merged_embeddings', {})
    if not merged_embeddings:
        common_widget_actions.create_and_show_messagebox(
            main_window, 'No Embeddings Found',
            'There are no embeddings to iterate over.',
            parent_widget=getattr(main_window, 'saveImageButton', None)
        )
        return

    # 顯示進度條（沿用你現有的 UI 元件）
    main_window.batchProgressBar.setValue(0)
    main_window.batchProgressLabel.setText("Starting embedding batch (current frame)...")
    main_window.batchProgressWidget.show()

    # 若上一次 worker 還在，就不要重複啟動
    worker = getattr(main_window, 'frame_emb_batch_worker', None)
    if worker and worker.isRunning():
        print("Embedding frame batch already running.")
        return

    # 建立並啟動 worker
    worker = CurrentFrameEmbBatchWorker(main_window)
    main_window.frame_emb_batch_worker = worker
    worker.progress.connect(main_window.update_batch_progress)
    worker.finished.connect(main_window.on_batch_processing_finished)
    worker.start()