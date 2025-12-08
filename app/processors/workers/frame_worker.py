import traceback
from typing import TYPE_CHECKING
import threading
import math
from math import floor, ceil
from PIL import Image
import torch
from skimage import transform as trans
import numpy as np
from torchvision.transforms import v2
import torchvision
from torchvision import transforms

import numpy
import cv2
import torch.nn.functional as F

from app.processors.utils import faceutil
import app.ui.widgets.actions.common_actions as common_widget_actions
from app.ui.widgets.actions import video_control_actions
from app.helpers.miscellaneous import ParametersDict, get_scaling_transforms

if TYPE_CHECKING:
    from app.ui.main_ui import MainWindow

torchvision.disable_beta_transforms_warning()
t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = None, None, None, None, None, None, None, None, None, None, None
class FrameWorker(threading.Thread):
    def __init__(
        self,
        frame,
        main_window: 'MainWindow',
        frame_number,
        frame_queue,
        is_single_frame=False,
        is_batch_processing=False,
        source_embedding_override=None,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.frame = frame
        self.main_window = main_window
        self.frame_number = frame_number
        self.models_processor = main_window.models_processor
        self.video_processor = main_window.video_processor
        self.is_single_frame = is_single_frame
        self.is_batch_processing = is_batch_processing
        self.parameters = {}
        self.target_faces = main_window.target_faces
        self.compare_images = []
        self.is_view_face_compare: bool = False
        self.is_view_face_mask: bool = False
        self.lock = threading.Lock()
        self.source_embedding_override = source_embedding_override
        self._active_source_embedding = None


    def _apply_embedding_override_if_any(self):
        """
        If a source embedding override is provided, set it as the active embedding.
        """
        try:
            override = self.source_embedding_override

            if override is None:
                emb_dict = getattr(self.main_window, "batch_processing_source_embedding_override", None)
                if isinstance(emb_dict, dict):
                    if "embedding_store" in emb_dict and isinstance(emb_dict["embedding_store"], dict):
                        emb_dict = emb_dict["embedding_store"]
                    model_name = self.main_window.control.get("RecognitionModelSelection")
                    override = emb_dict.get(model_name, None)

            if override is not None:
                self._active_source_embedding = np.asarray(override, dtype=np.float32)

                try:
                    setattr(self.main_window, "current_selected_embedding", self._active_source_embedding)
                except Exception:
                    pass
                try:
                    setattr(self.models_processor, "active_source_embedding", self._active_source_embedding)
                except Exception:
                    pass
        except Exception:
            traceback.print_exc()

    def run(self):
        self._apply_embedding_override_if_any()
        try:
            with self.main_window.models_processor.model_lock:
                video_control_actions.update_parameters_and_control_from_marker(self.main_window, self.frame_number)
            self.parameters = self.main_window.parameters.copy()
            self.is_view_face_compare = self.main_window.faceCompareCheckBox.isChecked() 
            self.is_view_face_mask = self.main_window.faceMaskCheckBox.isChecked() 

            if self.main_window.swapfacesButton.isChecked() or self.main_window.editFacesButton.isChecked() or self.main_window.control['FrameEnhancerEnableToggle']:
                self.frame = self.process_frame()
            else:
                self.frame = self.frame[..., ::-1]
            self.frame = np.ascontiguousarray(self.frame)

            if not self.is_batch_processing:
                pixmap = common_widget_actions.get_pixmap_from_frame(self.main_window, self.frame)

                if self.video_processor.file_type=='webcam' and not self.is_single_frame:
                    self.video_processor.webcam_frame_processed_signal.emit(pixmap, self.frame)
                elif not self.is_single_frame:
                    self.video_processor.frame_processed_signal.emit(self.frame_number, pixmap, self.frame)
                else:
                    self.video_processor.single_frame_processed_signal.emit(self.frame_number, pixmap, self.frame)

            self.frame_queue.get()
            self.frame_queue.task_done()

            if not self.is_batch_processing and self.video_processor.frame_queue.empty() and not self.video_processor.processing and self.video_processor.next_frame_to_display >= self.video_processor.max_frame_number:
                self.video_processor.stop_processing()

        except Exception as e:
            print(f"Error in FrameWorker: {e}")
            traceback.print_exc()
            
    def set_scaling_transforms(self, parameters):
        global t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift

        t512, t384, t256, t128, interpolation_get_cropped_face_kps, interpolation_original_face_128_384, interpolation_original_face_512, interpolation_Untransform, t256_face, interpolation_expression_faceeditor_back, interpolation_block_shift = get_scaling_transforms(parameters)    
    
    def tensor_to_pil(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)

        if tensor.dim() == 3 and tensor.shape[0] == 1:
            tensor = tensor.repeat(3, 1, 1)

        if tensor.dtype == torch.float32 or tensor.dtype == torch.float64:
            tensor = tensor.byte()
        
        tensor = tensor.permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(tensor)
        
    def process_frame(self):
        img = torch.from_numpy(self.frame.astype('uint8')).to(self.models_processor.device)
        img = img.permute(2,0,1)

        original_img_x = img.size()[2]
        original_img_y = img.size()[1]

        # Reverted to V5 scaling logic for stable processing
        if original_img_x < 512 and original_img_y < 512:
            if original_img_x <= original_img_y:
                new_height = int(512 * original_img_y / original_img_x)
                tscale = v2.Resize((new_height, 512), antialias=False)
            else:
                new_height = 512
                tscale = v2.Resize((new_height, int(512 * original_img_x / original_img_y)), antialias=False)
            img = tscale(img)
        elif original_img_x < 512:
            new_height = int(512 * original_img_y / original_img_x)
            tscale = v2.Resize((new_height, 512), antialias=False)
            img = tscale(img)
        elif original_img_y < 512:
            new_height = 512
            tscale = v2.Resize((new_height, int(512 * original_img_x / original_img_y)), antialias=False)
            img = tscale(img)

        control = self.main_window.control.copy()
        
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        use_landmark_detection=control['LandmarkDetectToggle']
        landmark_detect_mode=control['LandmarkDetectModelSelection']
        from_points = control["DetectFromPointsToggle"]
        if self.main_window.editFacesButton.isChecked():
            if not use_landmark_detection or landmark_detect_mode=="5":
                use_landmark_detection = True
                landmark_detect_mode = "203"
            from_points = True

        bboxes, kpss_5, kpss = self.models_processor.run_detect(img, control['DetectorModelSelection'], max_num=control['MaxFacesToDetectSlider'], score=control['DetectorScoreSlider']/100.0, input_size=(512, 512), use_landmark_detection=use_landmark_detection, landmark_detect_mode=landmark_detect_mode, landmark_score=control["LandmarkDetectScoreSlider"]/100.0, from_points=from_points, rotation_angles=[0] if not control["AutoRotationToggle"] else [0, 90, 180, 270])
        
        det_faces_data = []
        if len(kpss_5)>0:
            for i in range(kpss_5.shape[0]):
                face_emb, _ = self.models_processor.run_recognize_direct(img, kpss_5[i], control['SimilarityTypeSelection'], control['RecognitionModelSelection'])
                det_faces_data.append({
                    'kps_5': kpss_5[i], 
                    'kps_all': kpss[i], 
                    'embedding': face_emb, 
                    'bbox': bboxes[i],
                    'original_face': None,
                    'swap_mask': None
                })
        
        if self.main_window.editFacesButton.isChecked() and det_faces_data:
            for fface in det_faces_data:
                for _, target_face in self.target_faces.items():
                    parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)
                    sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                    if sim >= parameters['SimilarityThresholdSlider']:
                        if (parameters['FaceEditorEnableToggle'] or parameters['FaceMakeupEnableToggle'] or 
                            parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or 
                            parameters['LipsMakeupEnableToggle']):
                            self.set_scaling_transforms(parameters)
                            img = self.swap_edit_face_core(img, fface['kps_all'], parameters, control)

        if self.main_window.swapfacesButton.isChecked() and det_faces_data:
            is_override_active = self.is_batch_processing and self._active_source_embedding is not None
            if is_override_active:
                if self.target_faces:
                    ref_target_face = next(iter(self.target_faces.values()))
                    for fface in det_faces_data:
                        parameters = ParametersDict(self.parameters.get(ref_target_face.face_id, {}), self.main_window.default_parameters)
                        self.set_scaling_transforms(parameters)
                        
                        s_e = None
                        fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], parameters)
                        arcface_model = self.models_processor.get_arcface_model(parameters['SwapModelSelection'])
                        dfm_model = parameters['DFMModelSelection']
                        if parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                           s_e = self._active_source_embedding
                        
                        if s_e is not None and np.isnan(s_e).any():
                            s_e = None
                        
                        if s_e is not None:
                            t_e = fface['embedding']
                            img, fface['original_face'], fface['swap_mask'] = self.swap_core(img, fface['kps_5'], fface['kps_all'], s_e=s_e, t_e=t_e, parameters=parameters, control=control, dfm_model=dfm_model)

            else:
                for fface in det_faces_data:
                    for _, target_face in self.target_faces.items():
                        parameters = ParametersDict(self.parameters[target_face.face_id], self.main_window.default_parameters)
                        self.set_scaling_transforms(parameters)
                        sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                        if sim >= parameters['SimilarityThresholdSlider']:
                            s_e = None
                            fface['kps_5'] = self.keypoints_adjustments(fface['kps_5'], parameters)
                            arcface_model = self.models_processor.get_arcface_model(parameters['SwapModelSelection'])
                            dfm_model = parameters['DFMModelSelection']
                            
                            if parameters['SwapModelSelection'] != 'DeepFaceLive (DFM)':
                                s_e = target_face.assigned_input_embedding.get(arcface_model, None)

                            if s_e is not None and numpy.isnan(s_e).any():
                                s_e = None
                            
                            img, fface['original_face'], fface['swap_mask'] = self.swap_core(img, fface['kps_5'], fface['kps_all'], s_e=s_e, t_e=target_face.get_embedding(arcface_model), parameters=parameters, control=control, dfm_model=dfm_model)
        
        compare_mode = self.is_view_face_mask or self.is_view_face_compare
        
        if control['ManualRotationEnableToggle']:
            img = v2.functional.rotate(img, angle=-control['ManualRotationAngleSlider'], interpolation=v2.InterpolationMode.BILINEAR, expand=True)

        if control['ShowAllDetectedFacesBBoxToggle']:
            img = self.draw_bounding_boxes_on_detected_faces(img, det_faces_data, control)

        if control["ShowLandmarksEnableToggle"] and det_faces_data:
            img = img.permute(1,2,0)
            img = self.paint_face_landmarks(img, det_faces_data, control)
            img = img.permute(2,0,1)

        if compare_mode:
            img = self.get_compare_faces_image(img, det_faces_data, control)

        if control['FrameEnhancerEnableToggle'] and not compare_mode:
            img = self.enhance_core(img, control=control)
        # FIX: Added 'elif' to prevent downscaling after frame enhancement
        elif not compare_mode and (img.size()[2] != original_img_x or img.size()[1] != original_img_y):
            tscale_back = v2.Resize((original_img_y, original_img_x), antialias=False)
            img = tscale_back(img)
        
        img = img.permute(1,2,0)
        img = img.cpu().numpy()
        return img[..., ::-1]
        
    def keypoints_adjustments(self, kps_5: np.ndarray, parameters: dict) -> np.ndarray:
        if parameters['FaceAdjEnableToggle']:
            kps_5[:,0] += parameters['KpsXSlider']
            kps_5[:,1] += parameters['KpsYSlider']
            kps_5[:,0] -= 255
            kps_5[:,0] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,0] += 255
            kps_5[:,1] -= 255
            kps_5[:,1] *= (1+parameters['KpsScaleSlider']/100)
            kps_5[:,1] += 255

        if parameters['LandmarksPositionAdjEnableToggle']:
            kps_5[0][0] += parameters['EyeLeftXAmountSlider']
            kps_5[0][1] += parameters['EyeLeftYAmountSlider']
            kps_5[1][0] += parameters['EyeRightXAmountSlider']
            kps_5[1][1] += parameters['EyeRightYAmountSlider']
            kps_5[2][0] += parameters['NoseXAmountSlider']
            kps_5[2][1] += parameters['NoseYAmountSlider']
            kps_5[3][0] += parameters['MouthLeftXAmountSlider']
            kps_5[3][1] += parameters['MouthLeftYAmountSlider']
            kps_5[4][0] += parameters['MouthRightXAmountSlider']
            kps_5[4][1] += parameters['MouthRightYAmountSlider']
        return kps_5
    
    def paint_face_landmarks(self, img: torch.Tensor, det_faces_data: list, control: dict) -> torch.Tensor:
        p = 2
        for i, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id]
                sim = self.models_processor.findCosineDistance(fface['embedding'], target_face.get_embedding(control['RecognitionModelSelection']))
                if sim>=parameters['SimilarityThresholdSlider']:
                    if parameters['LandmarksPositionAdjEnableToggle']:
                        kcolor = tuple((255, 0, 0))
                        keypoints = fface['kps_5']
                    else:
                        kcolor = tuple((0, 255, 255))
                        keypoints = fface['kps_all']

                    for kpoint in keypoints:
                        for i in range(-1, p):
                            for j in range(-1, p):
                                try:
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][0] = kcolor[0]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][1] = kcolor[1]
                                    img[int(kpoint[1])+i][int(kpoint[0])+j][2] = kcolor[2]
                                except (ValueError, IndexError):
                                    continue
        return img
    
    def draw_bounding_boxes_on_detected_faces(self, img: torch.Tensor, det_faces_data: list, control: dict):
        for i, fface in enumerate(det_faces_data):
            color = [0, 255, 0]
            bbox = fface['bbox']
            x_min, y_min, x_max, y_max = map(int, bbox)
            _, h, w = img.shape
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w - 1, x_max), min(h - 1, y_max)
            max_dimension = max(img.shape[1], img.shape[2])
            thickness = max(4, max_dimension // 400)
            color_tensor = torch.tensor(color, dtype=img.dtype, device=img.device).view(-1, 1, 1)
            img[:, y_min:y_min + thickness, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_max - thickness + 1:y_max + 1, x_min:x_max + 1] = color_tensor.expand(-1, thickness, x_max - x_min + 1)
            img[:, y_min:y_max + 1, x_min:x_min + thickness] = color_tensor.expand(-1, y_max - y_min + 1, thickness)
            img[:, y_min:y_max + 1, x_max - thickness + 1:x_max + 1] = color_tensor.expand(-1, y_max - y_min + 1, thickness)   
        return img

    def get_compare_faces_image(self, img: torch.Tensor, det_faces_data: dict, control: dict) -> torch.Tensor:
        imgs_to_vstack = []
        for _, fface in enumerate(det_faces_data):
            for _, target_face in self.main_window.target_faces.items():
                parameters = self.parameters[target_face.face_id]
                sim = self.models_processor.findCosineDistance(
                    fface['embedding'], 
                    target_face.get_embedding(control['RecognitionModelSelection'])
                )
                if sim >= parameters['SimilarityThresholdSlider']:
                    modified_face = self.get_cropped_face_using_kps(img, fface['kps_5'], parameters)
                    if control['FrameEnhancerEnableToggle']:
                        modified_face_enhance = self.enhance_core(modified_face, control=control)
                        modified_face_enhance = modified_face_enhance.float() / 255.0
                        modified_face = torch.functional.F.interpolate(
                            modified_face_enhance.unsqueeze(0),
                            size=modified_face.shape[1:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                        modified_face = (modified_face * 255).clamp(0, 255).to(dtype=torch.uint8)
                    
                    imgs_to_cat = []
                    if fface.get('original_face') is not None:
                        imgs_to_cat.append(fface['original_face'].permute(2, 0, 1))
                    imgs_to_cat.append(modified_face)
                    if fface.get('swap_mask') is not None:
                        mask_view = 255 - fface['swap_mask']
                        imgs_to_cat.append(mask_view.permute(2, 0, 1))
  
                    img_compare = torch.cat(imgs_to_cat, dim=2)
                    imgs_to_vstack.append(img_compare)
    
        if imgs_to_vstack:
            max_width = max(img_to_stack.size(2) for img_to_stack in imgs_to_vstack)
            padded_imgs = [
                torch.nn.functional.pad(img_to_stack, (0, max_width - img_to_stack.size(2), 0, 0)) 
                for img_to_stack in imgs_to_vstack
            ]
            img_vstack = torch.cat(padded_imgs, dim=1)
            img = img_vstack
        return img
        
    def get_cropped_face_using_kps(self, img: torch.Tensor, kps_5: np.ndarray, parameters: dict) -> torch.Tensor:
        tform = self.get_face_similarity_tform(parameters['SwapModelSelection'], kps_5)
        face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=interpolation_get_cropped_face_kps)
        face_512 = v2.functional.crop(face_512, 0,0, 512, 512)
        return face_512

    def get_face_similarity_tform(self, swapper_model: str, kps_5: np.ndarray) -> trans.SimilarityTransform:
        tform = trans.SimilarityTransform()
        if swapper_model not in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3', 'CSCS'):
            dst = faceutil.get_arcface_template(image_size=512, mode='arcface128')
            dst = np.squeeze(dst)
            tform.estimate(kps_5, dst)
        elif swapper_model == "CSCS":
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            tform.estimate(kps_5, self.models_processor.FFHQ_kps)
        else:
            dst = faceutil.get_arcface_template(image_size=512, mode='arcfacemap')
            M, _ = faceutil.estimate_norm_arcface_template(kps_5, src=dst)
            tform.params[0:2] = M
        return tform
        
    def get_transformed_and_scaled_faces(self, tform, img):
        original_face_512 = v2.functional.affine(img, tform.rotation*57.2958, (tform.translation[0], tform.translation[1]) , tform.scale, 0, center = (0,0), interpolation=interpolation_original_face_512)
        original_face_512 = v2.functional.crop(original_face_512, 0,0, 512, 512)
        original_face_384 = t384(original_face_512)
        original_face_256 = t256(original_face_512)
        original_face_128 = t128(original_face_256)
        return original_face_512, original_face_384, original_face_256, original_face_128
    
    def get_affined_face_dim_and_swapping_latents(self, original_faces: tuple, swapper_model, dfm_model, s_e, t_e, parameters, tform):
        original_face_512, original_face_384, original_face_256, original_face_128 = original_faces
        if swapper_model == 'Inswapper128':
            self.models_processor.load_inswapper_iss_emap('Inswapper128')
            latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_inswapper_latent(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)

            dim = 1
            
            if parameters['SwapperResAutoSelectEnableToggle']:
                if tform.scale <= 1.25:
                    dim, input_face_affined = 4, original_face_512
                elif tform.scale <= 1.75:
                    dim, input_face_affined = 3, original_face_384
                elif tform.scale <= 2.25:
                    dim, input_face_affined = 2, original_face_256
                else:
                    dim, input_face_affined = 1, original_face_128
            else:
                res_map = {'128': (1, original_face_128), '256': (2, original_face_256), '384': (3, original_face_384), '512': (4, original_face_512)}
                dim, input_face_affined = res_map.get(parameters['SwapperResSelection'], (1, original_face_128))

        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            self.models_processor.load_inswapper_iss_emap(swapper_model)
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_iss(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim, input_face_affined = 2, original_face_256
        
        elif swapper_model in ('Hyperswap256 Version A', 'Hyperswap256 Version B', 'Hyperswap256 Version C'):
            version = swapper_model[-1]
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_hyperswap256(s_e, version)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_hyperswap256(t_e, version)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim, input_face_affined = 2, original_face_256

        elif swapper_model == 'SimSwap512':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_simswap512(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim, input_face_affined = 4, original_face_512

        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_ghost(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim, input_face_affined = 2, original_face_256

        elif swapper_model == 'CSCS':
            latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(s_e)).float().to(self.models_processor.device)
            if parameters['FaceLikenessEnableToggle'] and t_e is not None:
                factor = parameters['FaceLikenessFactorDecimalSlider']
                dst_latent = torch.from_numpy(self.models_processor.calc_swapper_latent_cscs(t_e)).float().to(self.models_processor.device)
                latent = latent - (factor * dst_latent)
            dim, input_face_affined = 2, original_face_256

        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            dfm_model = self.models_processor.load_dfm_model(dfm_model)
            latent = []
            dim, input_face_affined = 4, original_face_512
            
        return input_face_affined, dfm_model, dim, latent
    
    def get_swapped_and_prev_face(self, output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters):
        prev_face = input_face_affined.clone()
        if swapper_model == 'Inswapper128':
            with torch.no_grad():
                for _ in range(itex):
                    tiles = []
                    for j in range(dim):
                        for i in range(dim):
                            tile = input_face_affined[j::dim, i::dim]
                            tile = tile.permute(2, 0, 1)
                            tiles.append(tile)

                    idx = 0
                    for j in range(dim):
                        for i in range(dim):
                            input_tile = tiles[idx].unsqueeze(0).contiguous()
                            output_tile = torch.empty_like(input_tile)
                            self.models_processor.run_inswapper(input_tile, latent, output_tile)
                            output_tile = output_tile.squeeze(0).permute(1, 2, 0)
                            output[j::dim, i::dim] = output_tile.clone()
                            idx += 1

                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                output = torch.clamp(output * 255, 0, 255) 
                
        elif swapper_model in ('InStyleSwapper256 Version A', 'InStyleSwapper256 Version B', 'InStyleSwapper256 Version C'):
            version = swapper_model[-1]
            with torch.no_grad():
                for _ in range(itex):
                    input_face_disc = input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                    swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                    self.models_processor.run_iss_swapper(input_face_disc, latent, swapper_output, version)
                    swapper_output = swapper_output.squeeze(0).permute(1, 2, 0)
                    output = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255).clamp(0, 255)
        
        elif swapper_model in ('Hyperswap256 Version A', 'Hyperswap256 Version B', 'Hyperswap256 Version C'):
            version = swapper_model[-1]
            with torch.no_grad():
                for _ in range(itex):
                    input_face_disc = input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                    swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                    self.models_processor.run_hyperswap256(input_face_disc, latent, swapper_output, version)
                    swapper_output = swapper_output.squeeze(0).permute(1, 2, 0)
                    output = swapper_output.clone()
                    prev_face = input_face_affined.clone()
                    input_face_affined = output.clone()
                    output = torch.mul(output, 255).clamp(0, 255)

        elif swapper_model == 'SimSwap512':
            for _ in range(itex):
                input_face_disc = input_face_affined.permute(2, 0, 1).unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,512,512), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_simswap512(input_face_disc, latent, swapper_output)
                swapper_output = swapper_output.squeeze(0).permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                output = torch.mul(swapper_output, 255).clamp(0, 255)

        elif swapper_model in ('GhostFace-v1', 'GhostFace-v2', 'GhostFace-v3'):
            for _ in range(itex):
                input_face_disc = torch.sub(torch.div(torch.mul(input_face_affined, 255.0).permute(2, 0, 1).float(), 127.5), 1)
                input_face_disc = input_face_disc.unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_ghostface(input_face_disc, latent, swapper_output, swapper_model)
                swapper_output = torch.add(torch.mul(swapper_output[0].permute(1, 2, 0), 127.5), 127.5)
                prev_face = input_face_affined.clone()
                input_face_affined = torch.div(swapper_output.clone(), 255)
                output = swapper_output.clamp(0, 255)

        elif swapper_model == 'CSCS':
            for _ in range(itex):
                input_face_disc = v2.functional.normalize(input_face_affined.permute(2, 0, 1), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=False)
                input_face_disc = input_face_disc.unsqueeze(0).contiguous()
                swapper_output = torch.empty((1,3,256,256), dtype=torch.float32, device=self.models_processor.device).contiguous()
                self.models_processor.run_swapper_cscs(input_face_disc, latent, swapper_output)
                swapper_output = torch.add(torch.mul(swapper_output.squeeze(0), 0.5), 0.5).permute(1, 2, 0)
                prev_face = input_face_affined.clone()
                input_face_affined = swapper_output.clone()
                output = torch.mul(swapper_output, 255).clamp(0, 255)
        
        elif swapper_model == 'DeepFaceLive (DFM)' and dfm_model:
            out_celeb, _, _ = dfm_model.convert(original_face_512, parameters['DFMAmpMorphSlider']/100, rct=parameters['DFMRCTColorToggle'])
            prev_face = input_face_affined.clone()
            input_face_affined = out_celeb.clone()
            output = out_celeb.clone()

        output = output.permute(2, 0, 1)
        swap = t512(output)
        return swap, prev_face
    
    def get_border_mask(self, parameters):
        border_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device).unsqueeze(0)
        top = parameters['BorderTopSlider']
        left = parameters['BorderLeftSlider']
        right = 128 - parameters['BorderRightSlider']
        bottom = 128 - parameters['BorderBottomSlider']
        border_mask[:, :top, :] = 0
        border_mask[:, bottom:, :] = 0
        border_mask[:, :, :left] = 0
        border_mask[:, :, right:] = 0
        gauss = transforms.GaussianBlur(parameters['BorderBlurSlider']*2+1, (parameters['BorderBlurSlider']+1)*0.2)
        border_mask = gauss(border_mask)
        return border_mask

    def swap_core(self, img, kps_5, kps=False, s_e=None, t_e=None, parameters=None, control=None, dfm_model=False):
        s_e = s_e if isinstance(s_e, np.ndarray) else []
        t_e = t_e if isinstance(t_e, np.ndarray) else None
        parameters = parameters or {}
        control = control or {}
        swapper_model = parameters['SwapModelSelection']

        tform = self.get_face_similarity_tform(swapper_model, kps_5)
        t512_mask = v2.Resize((512, 512), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t256_mask = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
        t128_mask = v2.Resize((128, 128), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)

        original_face_512, original_face_384, original_face_256, original_face_128 = self.get_transformed_and_scaled_faces(tform, img)
        original_faces = (original_face_512, original_face_384, original_face_256, original_face_128)
        
        dim=1
        if (s_e is not None and len(s_e) > 0) or (swapper_model == 'DeepFaceLive (DFM)' and dfm_model):
            input_face_affined, dfm_model, dim, latent = self.get_affined_face_dim_and_swapping_latents(original_faces, swapper_model, dfm_model, s_e, t_e, parameters, tform)
            if parameters['FaceAdjEnableToggle']:
                input_face_affined = v2.functional.affine(input_face_affined, 0, (0, 0), 1 + parameters['FaceScaleAmountSlider'] / 100, 0, center=(dim*128/2, dim*128/2), interpolation=v2.InterpolationMode.BILINEAR)

            itex = ceil(parameters['StrengthAmountSlider'] / 100.) if parameters['StrengthEnableToggle'] else 1
            output_size = int(128 * dim)
            output = torch.zeros((output_size, output_size, 3), dtype=torch.float32, device=self.models_processor.device)
            input_face_affined = torch.div(input_face_affined.permute(1, 2, 0), 255.0)
            swap, prev_face = self.get_swapped_and_prev_face(output, input_face_affined, original_face_512, latent, itex, dim, swapper_model, dfm_model, parameters)
        else:
            swap = original_face_512
            if parameters['StrengthEnableToggle']:
                itex = ceil(parameters['StrengthAmountSlider'] / 100.)
                prev_face = torch.div(swap, 255.).permute(1, 2, 0)

        if parameters['StrengthEnableToggle']:
            if itex == 0:
                swap = original_face_512.clone()
            else:
                alpha = np.mod(parameters['StrengthAmountSlider'], 100) * 0.01
                if alpha == 0: alpha = 1
                prev_face = torch.mul(prev_face, 255).clamp(0, 255).permute(2, 0, 1)
                if dim != 4: prev_face = t512(prev_face)
                swap = torch.add(torch.mul(swap, alpha), torch.mul(prev_face, 1-alpha))

        border_mask = self.get_border_mask(parameters)
        swap_mask = torch.ones((128, 128), dtype=torch.float32, device=self.models_processor.device).unsqueeze(0)
        swap = torch.clamp(swap, 0.0, 255.0)
        swap_original = swap.clone()   
        
        if parameters["FaceRestorerEnableToggle"]:
            swap_autorestore = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider'])
        if parameters["FaceRestorerEnableToggle"] and (parameters["FaceRestorerAutoEnableToggle"] or parameters["FaceRestorerAutoMapEnableToggle"]):
            original_face_512_autorestore = original_face_512.clone()
            swap_mask_autorestore = t512_mask(swap_mask).clone()
            swap_mask_autorestore = (swap_mask_autorestore > 0.05).float()            
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            adjust_sharpness = float(parameters["FaceRestorerAutoAdjustSlider"])
            scale_factor = round(tform.scale, 2)
            alpha_auto, blur_value = self.face_restorer_auto(original_face_512_autorestore, swap_original, swap_autorestore, alpha_restorer, adjust_sharpness, scale_factor, swap_mask_autorestore)
        
        if parameters['FaceExpressionEnableToggle']:
            swap = self.apply_face_expression_restorer(original_face_512, swap, parameters)
        if parameters["OccluderEnableToggle"]:
            mask = self.models_processor.apply_occlusion(original_face_256, parameters["OccluderSizeSlider"])
            mask = t128_mask(mask)
            swap_mask = torch.mul(swap_mask, mask)
            gauss = transforms.GaussianBlur(parameters['OccluderXSegBlurSlider']*2+1, (parameters['OccluderXSegBlurSlider']+1)*0.2)
            swap_mask = gauss(swap_mask)

        if parameters["FaceParserEnableToggle"] or (parameters["XSegMouthEnableToggle"] and (parameters["DFLXSegSizeSlider"] != parameters["DFLXSeg2SizeSlider"])) or ((parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"]):
            mask, texture_mask, bg_mask, mouth = self.models_processor.apply_face_parser(swap, parameters, mode="swap")
            mask_original, texture_mask_original, bg_mask_original, mouth_original = self.models_processor.apply_face_parser(original_face_512, parameters, mode="original")
            if parameters["FaceParserEnableToggle"]:
                mask = torch.minimum(mask, mask_original)
                mask = t128_mask(mask)
                swap_mask = torch.mul(swap_mask, mask)
            if (parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"]:
                texture_mask = 1 - torch.clamp(texture_mask + bg_mask_original, 0, 1)
                texture_mask_original = 1 - torch.clamp(texture_mask_original + bg_mask_original, 0, 1)
                texture_mask = torch.minimum(texture_mask, texture_mask_original)            
                texture_mask = t512_mask(texture_mask)
                adjusted_mask = torch.add(texture_mask, (parameters['FaceParserBlendTextureSlider']/100)).clamp(0, 1)
             
        if parameters["DFLXSegEnableToggle"]:
            mouth = torch.max(t256_mask(mouth), t256_mask(mouth_original)) if parameters["XSegMouthEnableToggle"] and parameters["DFLXSegSizeSlider"] != parameters["DFLXSeg2SizeSlider"] else 0
            img_mask = self.models_processor.apply_dfl_xseg(original_face_256, -parameters["DFLXSegSizeSlider"], mouth, parameters)
            img_mask = t128_mask(img_mask)
            swap_mask = torch.mul(swap_mask, 1 - img_mask)

        if parameters["ClipEnableToggle"]:
            mask = self.models_processor.run_CLIPs(original_face_512, parameters["ClipText"], parameters["ClipAmountSlider"])
            mask = t128_mask(mask)
            swap_mask *= mask

        if parameters['RestoreMouthEnableToggle'] or parameters['RestoreEyesEnableToggle']:
            M = tform.params[0:2]
            homogeneous_kps = np.hstack([kps_5, np.ones((kps_5.shape[0], 1), dtype=np.float32)])
            dst_kps_5 = np.dot(homogeneous_kps, M.T)
            img_swap_mask = torch.ones((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            img_orig_mask = torch.zeros((1, 512, 512), dtype=torch.float32, device=self.models_processor.device).contiguous()
            if parameters['RestoreMouthEnableToggle']:
                img_swap_mask = self.models_processor.restore_mouth(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreMouthBlendAmountSlider']/100, parameters['RestoreMouthFeatherBlendSlider'], parameters['RestoreMouthSizeFactorSlider']/100, parameters['RestoreXMouthRadiusFactorDecimalSlider'], parameters['RestoreYMouthRadiusFactorDecimalSlider'], parameters['RestoreXMouthOffsetSlider'], parameters['RestoreYMouthOffsetSlider']).clamp(0, 1)
            if parameters['RestoreEyesEnableToggle']:
                img_swap_mask = self.models_processor.restore_eyes(img_orig_mask, img_swap_mask, dst_kps_5, parameters['RestoreEyesBlendAmountSlider']/100, parameters['RestoreEyesFeatherBlendSlider'], parameters['RestoreEyesSizeFactorDecimalSlider'],  parameters['RestoreXEyesRadiusFactorDecimalSlider'], parameters['RestoreYEyesRadiusFactorDecimalSlider'], parameters['RestoreXEyesOffsetSlider'], parameters['RestoreYEyesOffsetSlider'], parameters['RestoreEyesSpacingOffsetSlider']).clamp(0, 1)
            gauss = transforms.GaussianBlur(parameters['RestoreEyesMouthBlurSlider']*2+1, (parameters['RestoreEyesMouthBlurSlider']+1)*0.2)
            img_swap_mask = gauss(img_swap_mask)
            swap_mask = torch.mul(swap_mask, t128_mask(img_swap_mask))
        
        IM512 = tform.inverse.params[0:2, :]
        corners = np.array([[0,0], [0,511], [511, 0], [511, 511]])
        x = (IM512[0][0]*corners[:,0] + IM512[0][1]*corners[:,1] + IM512[0][2])
        y = (IM512[1][0]*corners[:,0] + IM512[1][1]*corners[:,1] + IM512[1][2])
        left, top = max(0, floor(np.min(x))), max(0, floor(np.min(y)))
        right, bottom = min(img.shape[2], ceil(np.max(x))), min(img.shape[1], ceil(np.max(y)))

        swap_backup = swap.clone()   

        if parameters["FaceRestorerEnableToggle"]:
            swap = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'], parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider'])
        if parameters["FaceRestorerEnableToggle"] and parameters["FaceRestorerAutoMapEnableToggle"]:    
            if blur_value != 0:
                kernel_size = 2 * blur_value + 1
                sigma = blur_value * 0.2
                swap_backup = transforms.GaussianBlur(kernel_size, sigma)(swap_backup) 
            swap_mask_autorestore = t512_mask(swap_mask).clone() > 0.0
            swap, _ = self.face_restorer_auto(original_face_512.clone(), swap_backup, swap, alpha_auto, float(parameters["FaceRestorerAutoAdjustSlider"]), round(tform.scale, 2), swap_mask_autorestore, parameters["FaceRestorerAutoAdjustKernelSlider"], pixelwise=True)
            swap = torch.where(swap_mask_autorestore, swap, original_face_512)
        elif parameters["FaceRestorerAutoEnableToggle"] and parameters["FaceRestorerEnableToggle"] and not parameters["FaceRestorerAutoMapEnableToggle"]:
            if blur_value != 0:
                swap = transforms.GaussianBlur(2 * blur_value + 1, blur_value * 0.2)(swap_backup) 
            elif alpha_auto != 0:
                swap = swap * alpha_auto + swap_backup * (1 - alpha_auto)
            else:
                swap = swap_backup 
        elif parameters["FaceRestorerEnableToggle"]:
            alpha_restorer = float(parameters["FaceRestorerBlendSlider"])/100.0
            swap = torch.add(torch.mul(swap, alpha_restorer), torch.mul(swap_backup, 1 - alpha_restorer))                             

        swap_backup = swap.clone()

        if parameters["TransferTextureEnableToggle"]:
            swap_mask_texture = t512_mask(swap_mask) > 0.0
            gradient_texture = self.gradient_magnitude(original_face_512, swap_mask_texture, 3, parameters['TransferTextureWeightSlider'], parameters['TransferTextureSigmaDecimalSlider'], 3, 0.5, 2, parameters['TransferTextureThetaSlider'], 1)
            gradient_texture *= (parameters['TransferTextureBlendAmountSlider']/50)
            swap += gradient_texture
            swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, swap_mask_texture, 100)
            if parameters["ExcludeMaskEnableToggle"]:
                swap_backup = faceutil.histogram_matching_DFL_Orig(original_face_512, swap_backup, swap_mask_texture, 100)
            swap = swap.clamp(0, 255)
            
        if parameters["AutoColorEnableToggle"]:
            swap_mask_autocolor = t512_mask(swap_mask).clone() > 0
            swap = torch.where(swap_mask_autocolor, swap, original_face_512)
            if parameters['AutoColorTransferTypeSelection'] == 'Test':
                swap = faceutil.histogram_matching(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'Test_Mask':
                swap = faceutil.histogram_matching_withmask(original_face_512, swap, swap_mask_autocolor, parameters["AutoColorBlendAmountSlider"])
                if parameters["ExcludeMaskEnableToggle"]:
                    swap_backup = faceutil.histogram_matching_withmask(original_face_512, swap_backup, swap_mask_autocolor, parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Test':
                swap = faceutil.histogram_matching_DFL_test(original_face_512, swap, parameters["AutoColorBlendAmountSlider"])
            elif parameters['AutoColorTransferTypeSelection'] == 'DFL_Orig':
                swap = faceutil.histogram_matching_DFL_Orig(original_face_512, swap, t512_mask(swap_mask), parameters["AutoColorBlendAmountSlider"])

        if parameters["DifferencingEnableToggle"]:
            swap_mask_diff = t512_mask(swap_mask) > 0
            swap = torch.where(swap_mask_diff, swap, original_face_512)
            mask = self.models_processor.apply_fake_diff(swap, original_face_512, parameters['DifferencingLowerLimitThreshSlider']/100, parameters['DifferencingLowerLimitValueSlider']/100, parameters['DifferencingUpperLimitThreshSlider']/100, parameters['DifferencingUpperLimitValueSlider']/100, parameters['DifferencingMiddleLimitValueSlider']/100)
            gauss = transforms.GaussianBlur(parameters['DifferencingBlendAmountSlider']*2+1, (parameters['DifferencingBlendAmountSlider']+1)*0.2)
            mask = gauss(mask.type(torch.float32))
            swap = (swap * mask + original_face_512 * (1-mask)).clamp(0, 255)
        
        if (parameters["TransferTextureEnableToggle"] or parameters["DifferencingEnableToggle"]) and parameters["ExcludeMaskEnableToggle"]:   
            swap = torch.add(torch.mul(swap, adjusted_mask), torch.mul(swap_backup, 1 - adjusted_mask)).clamp(0, 255)

        if parameters['ColorEnableToggle']:
            swap = v2.functional.adjust_gamma(swap.unsqueeze(0), parameters['ColorGammaDecimalSlider'], 1.0).squeeze(0)
            swap = swap.permute(1, 2, 0).float()
            del_color = torch.tensor([parameters['ColorRedSlider'], parameters['ColorGreenSlider'], parameters['ColorBlueSlider']], device=self.models_processor.device)
            swap = (swap + del_color).clamp(min=0., max=255.).permute(2, 0, 1).byte()
            swap = v2.functional.adjust_brightness(swap, parameters['ColorBrightnessDecimalSlider'])
            swap = v2.functional.adjust_contrast(swap, parameters['ColorContrastDecimalSlider'])
            swap = v2.functional.adjust_saturation(swap, parameters['ColorSaturationDecimalSlider'])
            swap = v2.functional.adjust_sharpness(swap, parameters['ColorSharpnessDecimalSlider'])
            swap = v2.functional.adjust_hue(swap, parameters['ColorHueDecimalSlider'])
        
        if parameters["FaceRestorerEnable2Toggle"]:
            swap2 = self.models_processor.apply_facerestorer(swap, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'], parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider'])
            alpha_restorer2 = float(parameters["FaceRestorerBlend2Slider"])/100.0
            swap = torch.add(torch.mul(swap2, alpha_restorer2), torch.mul(swap.float(), 1 - alpha_restorer2))

        if parameters['FinalBlendAdjEnableToggle'] and parameters['FinalBlendAmountSlider'] > 0:
            final_blur_strength = parameters['FinalBlendAmountSlider']
            gaussian_blur = transforms.GaussianBlur(kernel_size=2 * final_blur_strength + 1, sigma=final_blur_strength * 0.1)
            swap = gaussian_blur(swap)

        if parameters['ColorNoiseDecimalSlider'] > 0:
            noise = (torch.rand_like(swap.float()) - 0.5) * 2 * parameters['ColorNoiseDecimalSlider']
            swap = (swap.float() + noise).clamp(0.0, 255.0)

        if parameters["BlockShiftEnableToggle"]:
            tform_scale = min(8, round(parameters["BlockShiftAmountSlider"] * tform.scale/2))
            swap2 = self.apply_block_shift_gpu(swap, tform_scale, parameters["BlockShiftMaxAmountSlider"])        
            block_shift_blend = parameters["BlockShiftBlendAmountSlider"]/100.0
            swap = torch.add(torch.mul(swap2, block_shift_blend), torch.mul(swap.float(), 1 - block_shift_blend))                          
            
        if parameters['JPEGCompressionEnableToggle']:
            try:
                swap = faceutil.jpegBlur(swap, parameters["JPEGCompressionAmountSlider"])
            except Exception: pass

        gauss = transforms.GaussianBlur(parameters['OverallMaskBlendAmountSlider'] * 2 + 1, (parameters['OverallMaskBlendAmountSlider'] + 1) * 0.2)
        swap_mask = gauss(swap_mask)
        swap_mask = torch.mul(swap_mask, border_mask)
        swap_mask = t512_mask(swap_mask)
        swap = torch.mul(swap.float(), swap_mask)

        original_face_512_clone = original_face_512.clone().byte().permute(1, 2, 0) if self.is_view_face_compare else None
        swap_mask_clone = None
        if self.is_view_face_mask:
            swap_mask_clone = swap_mask.clone()
            swap_mask_clone = torch.sub(1, swap_mask_clone)
            swap_mask_clone = torch.cat((swap_mask_clone,swap_mask_clone,swap_mask_clone),0)
            swap_mask_clone = swap_mask_clone.permute(1, 2, 0)
            swap_mask_clone = torch.mul(swap_mask_clone, 255.).type(torch.uint8)

        pad_w, pad_h = img.shape[2] - 512, img.shape[1] - 512
        swap = v2.functional.pad(swap, (0, 0, pad_w, pad_h))
        swap = v2.functional.affine(swap, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=interpolation_Untransform, center = (0,0))
        swap = swap[0:3, top:bottom, left:right]

        swap_mask = v2.functional.pad(swap_mask, (0, 0, pad_w, pad_h))
        swap_mask = v2.functional.affine(swap_mask, tform.inverse.rotation*57.2958, (tform.inverse.translation[0], tform.inverse.translation[1]), tform.inverse.scale, 0, interpolation=v2.InterpolationMode.BILINEAR, center = (0,0))
        swap_mask = swap_mask[0:1, top:bottom, left:right]
        swap_mask_minus = torch.sub(1, swap_mask)

        img_crop = torch.mul(swap_mask_minus, img[0:3, top:bottom, left:right].float())
        swap = torch.add(swap, img_crop).byte().clamp(0, 255)
        img[0:3, top:bottom, left:right] = swap

        return img, original_face_512_clone, swap_mask_clone
        
    def enhance_core(self, img, control):
        enhancer_type = control['FrameEnhancerTypeSelection']
        match enhancer_type:
            case 'RealEsrgan-x2-Plus' | 'RealEsrgan-x4-Plus' | 'BSRGan-x2' | 'BSRGan-x4' | 'UltraSharp-x4' | 'UltraMix-x4' | 'RealEsr-General-x4v3':
                tile_size = 512
                if enhancer_type == 'RealEsrgan-x2-Plus' or enhancer_type == 'BSRGan-x2':
                    scale = 2
                else:
                    scale = 4
                image = img.type(torch.float32)
                if torch.max(image) > 256:
                    max_range = 65535
                else:
                    max_range = 255
                image = torch.div(image, max_range)
                image = torch.unsqueeze(image, 0).contiguous()
                image = self.models_processor.run_enhance_frame_tile_process(image, enhancer_type, tile_size=tile_size, scale=scale)
                image = torch.squeeze(image)
                image = torch.clamp(image, 0, 1)
                image = torch.mul(image, max_range)
                alpha = float(control["FrameEnhancerBlendSlider"])/100.0
                t_scale = v2.Resize((img.shape[1] * scale, img.shape[2] * scale), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                img = t_scale(img)
                img = torch.add(torch.mul(image, alpha), torch.mul(img, 1-alpha))
                if max_range == 255:
                    img = img.type(torch.uint8)
                else:
                    img = img.type(torch.uint16)

            case 'DeOldify-Artistic' | 'DeOldify-Stable' | 'DeOldify-Video':
                render_factor = 384
                _, h, w = img.shape
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)
                image = image.type(torch.float32)
                image = torch.unsqueeze(image, 0).contiguous()
                output = torch.empty((image.shape), dtype=torch.float32, device=self.models_processor.device).contiguous()
                match enhancer_type:
                    case 'DeOldify-Artistic':
                        self.models_processor.run_deoldify_artistic(image, output)
                    case 'DeOldify-Stable':
                        self.models_processor.run_deoldify_stable(image, output)
                    case 'DeOldify-Video':
                        self.models_processor.run_deoldify_video(image, output)
                output = torch.squeeze(output)
                t_resize_o = v2.Resize((h, w), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output = t_resize_o(output)
                output = faceutil.rgb_to_yuv(output, True)
                hires = faceutil.rgb_to_yuv(img, True)
                hires[1:3, :, :] = output[1:3, :, :]
                hires = faceutil.yuv_to_rgb(hires, True)
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                img = torch.add(torch.mul(hires, alpha), torch.mul(img, 1-alpha))
                img = img.type(torch.uint8)

            case 'DDColor-Artistic' | 'DDColor':
                render_factor = 384
                orig_l = faceutil.rgb_to_lab(img, True)
                orig_l = orig_l[0:1, :, :]
                t_resize_i = v2.Resize((render_factor, render_factor), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                image = t_resize_i(img)
                img_l = faceutil.rgb_to_lab(image, True)
                img_l = img_l[0:1, :, :]
                img_gray_lab = torch.cat((img_l, torch.zeros_like(img_l), torch.zeros_like(img_l)), dim=0)
                img_gray_rgb = faceutil.lab_to_rgb(img_gray_lab)
                tensor_gray_rgb = torch.unsqueeze(img_gray_rgb.type(torch.float32), 0).contiguous()
                output_ab = torch.empty((1, 2, render_factor, render_factor), dtype=torch.float32, device=self.models_processor.device)
                match enhancer_type:
                    case 'DDColor-Artistic':
                        self.models_processor.run_ddcolor_artistic(tensor_gray_rgb, output_ab)
                    case 'DDColor':
                        self.models_processor.run_ddcolor(tensor_gray_rgb, output_ab)
                output_ab = output_ab.squeeze(0)
                t_resize_o = v2.Resize((img.size(1), img.size(2)), interpolation=v2.InterpolationMode.BILINEAR, antialias=False)
                output_lab_resize = t_resize_o(output_ab)
                output_lab = torch.cat((orig_l, output_lab_resize), dim=0)
                output_rgb = faceutil.lab_to_rgb(output_lab, True)
                alpha = float(control["FrameEnhancerBlendSlider"]) / 100.0
                blended_img = torch.add(torch.mul(output_rgb, alpha), torch.mul(img, 1 - alpha))
                img = blended_img.type(torch.uint8)
        return img

    def apply_face_expression_restorer(self, driving, target, parameters):
        _, driving_lmk_crop, _ = self.models_processor.run_detect_landmark(driving, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        driving_face_256 = t256_face(driving.clone())
        c_d_eyes_lst = faceutil.calc_eye_close_ratio(driving_lmk_crop[None])
        c_d_lip_lst = faceutil.calc_lip_close_ratio(driving_lmk_crop[None])
        x_d_i_info = self.models_processor.lp_motion_extractor(driving_face_256, 'Human-Face')
        R_d_i = faceutil.get_rotation_matrix(x_d_i_info['pitch'], x_d_i_info['yaw'], x_d_i_info['roll'])
        
        driving_multiplier=parameters['FaceExpressionFriendlyFactorDecimalSlider']
        animation_region = parameters['FaceExpressionAnimationRegionSelection']
        if animation_region == 'all': animation_region = 'eyes,lips'

        flag_normalize_lip = parameters['FaceExpressionNormalizeLipsEnableToggle']
        lip_normalize_threshold = parameters['FaceExpressionNormalizeLipsThresholdDecimalSlider']
        flag_eye_retargeting = parameters['FaceExpressionRetargetingEyesEnableToggle']
        eye_retargeting_multiplier = parameters['FaceExpressionRetargetingEyesMultiplierDecimalSlider']
        flag_lip_retargeting = parameters['FaceExpressionRetargetingLipsEnableToggle']
        lip_retargeting_multiplier = parameters['FaceExpressionRetargetingLipsMultiplierDecimalSlider']
        
        target = torch.clamp(target, 0, 255).byte()
        _, source_lmk, _ = self.models_processor.run_detect_landmark(target, bbox=np.array([0, 0, 512, 512]), det_kpss=[], detect_mode='203', score=0.5, from_points=False)
        target_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(target, source_lmk, dsize=512, scale=parameters['FaceExpressionCropScaleDecimalSlider'], vy_ratio=parameters['FaceExpressionVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)
        target_face_256 = t256_face(target_face_512)

        x_s_info = self.models_processor.lp_motion_extractor(target_face_256, 'Human-Face')
        x_c_s = x_s_info['kp']
        R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
        f_s = self.models_processor.lp_appearance_feature_extractor(target_face_256, 'Human-Face')
        x_s = faceutil.transform_keypoint(x_s_info)

        lip_delta_before_animation = None
        if flag_normalize_lip and source_lmk is not None:
            combined_lip_ratio_tensor = faceutil.calc_combined_lip_ratio([0.], source_lmk, device=self.models_processor.device)
            if combined_lip_ratio_tensor[0][0] >= lip_normalize_threshold:
                lip_delta_before_animation = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio_tensor)

        delta_new = x_s_info['exp'].clone()
        R_new = (R_d_i @ R_d_i.permute(0, 2, 1)) @ R_s if "pose" in animation_region else R_s
        if "exp" in animation_region or "all" in animation_region:
            delta_new = x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device))
        else:
            if "lips" in animation_region:
                for i in [6, 12, 14, 17, 19, 20]: delta_new[:, i, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - torch.from_numpy(self.models_processor.lp_lip_array).to(dtype=torch.float32, device=self.models_processor.device)))[:, i, :]
            if "eyes" in animation_region:
                for i in [11, 13, 15, 16, 18]: delta_new[:, i, :] = (x_s_info['exp'] + (x_d_i_info['exp'] - 0))[:, i, :]
        
        scale_new, t_new = x_s_info['scale'], x_s_info['t']
        t_new[..., 2].fill_(0)
        x_d_i_new = scale_new * (x_c_s @ R_new + delta_new) + t_new
        
        if flag_eye_retargeting or flag_lip_retargeting:
            eyes_delta, lip_delta = None, None
            if flag_eye_retargeting and source_lmk is not None:
                combined_eye_ratio = faceutil.calc_combined_eye_ratio(c_d_eyes_lst, source_lmk, device=self.models_processor.device) * eye_retargeting_multiplier
                eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined_eye_ratio, parameters["FaceEditorTypeSelection"])
            if flag_lip_retargeting and source_lmk is not None:
                combined_lip_ratio = faceutil.calc_combined_lip_ratio(c_d_lip_lst, source_lmk, device=self.models_processor.device) * lip_retargeting_multiplier
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined_lip_ratio, parameters["FaceEditorTypeSelection"])
            x_d_i_new = x_s + (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)
            x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])
        else:
            if lip_delta_before_animation is not None:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"]) + lip_delta_before_animation
            else:
                x_d_i_new = self.models_processor.lp_stitching(x_s, x_d_i_new, parameters["FaceEditorTypeSelection"])

        x_d_i_new = x_s + (x_d_i_new - x_s) * driving_multiplier
        out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_i_new, parameters["FaceEditorTypeSelection"]).squeeze(0).clamp(0, 1)

        t = trans.SimilarityTransform()
        t.params[0:2] = M_c2o
        dsize = (target.shape[1], target.shape[2])
        out = faceutil.pad_image_by_size(out, dsize)
        out = v2.functional.affine(out, t.rotation*57.2958, translate=(t.translation[0], t.translation[1]), scale=t.scale, shear=(0.0, 0.0), interpolation=interpolation_expression_faceeditor_back, center=(0, 0))
        out = v2.functional.crop(out, 0, 0, dsize[1], dsize[0])
        return torch.mul(out, 255.0).clamp(0, 255).float()

    def swap_edit_face_core(self, img, kps, parameters, control, **kwargs):
        _, lmk_crop, _ = self.models_processor.run_detect_landmark(img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)
        
        original_face_512, M_o2c, M_c2o = faceutil.warp_face_by_face_landmark_x(
            img, lmk_crop, dsize=512,
            scale=parameters["FaceEditorCropScaleDecimalSlider"],
            vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'],
            interpolation=interpolation_expression_faceeditor_back
        )

        if not self.main_window.swapfacesButton.isChecked():
            restorer_applied = False
            if parameters['FaceRestorerEnableToggle']:
                restorer_applied = True
                face_backup = original_face_512.clone()
                restored_face = self.models_processor.apply_facerestorer(
                    face_backup, parameters['FaceRestorerDetTypeSelection'], parameters['FaceRestorerTypeSelection'],
                    parameters["FaceRestorerBlendSlider"], parameters['FaceFidelityWeightDecimalSlider'], control['DetectorScoreSlider']
                )
                alpha = float(parameters["FaceRestorerBlendSlider"]) / 100.0
                original_face_512 = torch.add(torch.mul(restored_face, alpha), torch.mul(face_backup.float(), 1 - alpha))

            if parameters['FaceRestorerEnable2Toggle']:
                if not restorer_applied: original_face_512 = original_face_512.float()
                restorer_applied = True
                face_backup = original_face_512.clone()
                restored_face_2 = self.models_processor.apply_facerestorer(
                    face_backup, parameters['FaceRestorerDetType2Selection'], parameters['FaceRestorerType2Selection'],
                    parameters["FaceRestorerBlend2Slider"], parameters['FaceFidelityWeight2DecimalSlider'], control['DetectorScoreSlider']
                )
                alpha_2 = float(parameters["FaceRestorerBlend2Slider"]) / 100.0
                original_face_512 = torch.add(torch.mul(restored_face_2, alpha_2), torch.mul(face_backup, 1 - alpha_2))
            
            if restorer_applied:
                original_face_512 = original_face_512.clamp(0, 255).byte()

            if control['FrameEnhancerEnableToggle']:
                original_face_512 = self.enhance_core(original_face_512, control)

        if parameters['FaceEditorEnableToggle']:
            source_eye_ratio = faceutil.calc_eye_close_ratio(lmk_crop[None])
            source_lip_ratio = faceutil.calc_lip_close_ratio(lmk_crop[None])
            init_source_eye_ratio = round(float(source_eye_ratio.mean()), 2)
            init_source_lip_ratio = round(float(source_lip_ratio[0][0]), 2)
            original_face_256 = t256_face(original_face_512)

            x_s_info = self.models_processor.lp_motion_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_d_pitch = x_s_info['pitch'] + parameters['HeadPitchSlider']
            x_d_yaw = x_s_info['yaw'] + parameters['HeadYawSlider']
            x_d_roll = x_s_info['roll'] + parameters['HeadRollSlider']
            R_s = faceutil.get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
            R_d = faceutil.get_rotation_matrix(x_d_pitch, x_d_yaw, x_d_roll)
            f_s = self.models_processor.lp_appearance_feature_extractor(original_face_256, parameters["FaceEditorTypeSelection"])
            x_s = faceutil.transform_keypoint(x_s_info)

            mov_x, mov_y, mov_z = torch.tensor(parameters['XAxisMovementDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['YAxisMovementDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['ZAxisMovementDecimalSlider']).to(self.models_processor.device)
            eye_x, eye_y = torch.tensor(parameters['EyeGazeHorizontalDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['EyeGazeVerticalDecimalSlider']).to(self.models_processor.device)
            smile, wink, eyebrow = torch.tensor(parameters['MouthSmileDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['EyeWinkDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['EyeBrowsDirectionDecimalSlider']).to(self.models_processor.device)
            lip_v0, lip_v1, lip_v2, lip_v3 = torch.tensor(parameters['MouthPoutingDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['MouthPursingDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['MouthGrinDecimalSlider']).to(self.models_processor.device), torch.tensor(parameters['LipsCloseOpenSlider']).to(self.models_processor.device)

            x_c_s, delta_new, scale_new, t_new = x_s_info['kp'], x_s_info['exp'], x_s_info['scale'], x_s_info['t']
            R_d_new = (R_d @ R_s.permute(0, 2, 1)) @ R_s

            if eye_x != 0 or eye_y != 0: delta_new = faceutil.update_delta_new_eyeball_direction(eye_x, eye_y, delta_new)
            if smile != 0: delta_new = faceutil.update_delta_new_smile(smile, delta_new)
            if wink != 0: delta_new = faceutil.update_delta_new_wink(wink, delta_new)
            if eyebrow != 0: delta_new = faceutil.update_delta_new_eyebrow(eyebrow, delta_new)
            if lip_v0 != 0: delta_new = faceutil.update_delta_new_lip_variation_zero(lip_v0, delta_new)
            if lip_v1 != 0: delta_new = faceutil.update_delta_new_lip_variation_one(lip_v1, delta_new)
            if lip_v2 != 0: delta_new = faceutil.update_delta_new_lip_variation_two(lip_v2, delta_new)
            if lip_v3 != 0: delta_new = faceutil.update_delta_new_lip_variation_three(lip_v3, delta_new)
            if mov_x != 0: delta_new = faceutil.update_delta_new_mov_x(-mov_x, delta_new)
            if mov_y != 0: delta_new = faceutil.update_delta_new_mov_y(mov_y, delta_new)

            x_d_new = mov_z * scale_new * (x_c_s @ R_d_new + delta_new) + t_new
            eyes_delta, lip_delta = None, None

            eye_ratio = max(min(init_source_eye_ratio + parameters['EyesOpenRatioDecimalSlider'], 0.80), 0.00)
            if eye_ratio != init_source_eye_ratio:
                combined = faceutil.calc_combined_eye_ratio([[float(eye_ratio)]], lmk_crop, device=self.models_processor.device)
                eyes_delta = self.models_processor.lp_retarget_eye(x_s, combined, parameters["FaceEditorTypeSelection"])

            lip_ratio = max(min(init_source_lip_ratio + parameters['LipsOpenRatioDecimalSlider'], 0.80), 0.00)
            if lip_ratio != init_source_lip_ratio:
                combined = faceutil.calc_combined_lip_ratio([[float(lip_ratio)]], lmk_crop, device=self.models_processor.device)
                lip_delta = self.models_processor.lp_retarget_lip(x_s, combined, parameters["FaceEditorTypeSelection"])

            x_d_new += (eyes_delta if eyes_delta is not None else 0) + (lip_delta if lip_delta is not None else 0)

            if kwargs.get('flag_stitching_retargeting_input', True):
                x_d_new = self.models_processor.lp_stitching(x_s, x_d_new, parameters["FaceEditorTypeSelection"])

            out = self.models_processor.lp_warp_decode(f_s, x_s, x_d_new, parameters["FaceEditorTypeSelection"]).squeeze(0).clamp(0, 1)
            gauss = transforms.GaussianBlur(parameters['FaceEditorBlurAmountSlider']*2+1, (parameters['FaceEditorBlurAmountSlider']+1)*0.2)
            mask_crop = gauss(self.models_processor.lp_mask_crop)
            img = faceutil.paste_back_adv(out, M_c2o, img, mask_crop)

        if parameters['FaceMakeupEnableToggle'] or parameters['HairMakeupEnableToggle'] or parameters['EyeBrowsMakeupEnableToggle'] or parameters['LipsMakeupEnableToggle']:
            _, makeup_lmk_crop, _ = self.models_processor.run_detect_landmark(img, bbox=[], det_kpss=kps, detect_mode='203', score=0.5, from_points=True)
            makeup_face_512, _, makeup_M_c2o = faceutil.warp_face_by_face_landmark_x(img, makeup_lmk_crop, dsize=512, scale=parameters['FaceEditorCropScaleDecimalSlider'], vy_ratio=parameters['FaceEditorVYRatioDecimalSlider'], interpolation=interpolation_expression_faceeditor_back)
            out, _ = self.models_processor.apply_face_makeup(makeup_face_512, parameters)
            gauss = transforms.GaussianBlur(11, 1.2)
            out = torch.clamp(torch.div(out, 255.0), 0, 1).float()
            mask_crop = gauss(self.models_processor.lp_mask_crop)
            img = faceutil.paste_back_adv(out, makeup_M_c2o, img, mask_crop)

        return img

    def gradient_magnitude(self, image, mask, kernel_size, weighting_strength, sigma, lambd, gamma, psi, theta_count, hoch):
        image = image.float()
        kernel_size = max(1, 2 * kernel_size - 1)
        theta_values = torch.linspace(0, math.pi, theta_count, device=image.device)
        magnitude = self.apply_gabor_filter_torch(image, kernel_size, sigma, lambd, gamma, psi, theta_values)
        inverted = magnitude.amax(dim=(1, 2), keepdim=True) - magnitude
        if weighting_strength > 0:
            intensity_weight = ((image * mask) / 255.0) ** hoch
            return inverted * ((1 - weighting_strength) + weighting_strength * intensity_weight)
        return inverted

    def apply_gabor_filter_torch(self, image, kernel_size, sigma, lambd, gamma, psi, theta_values):
        C, H, W = image.shape
        image = image.unsqueeze(0)
        kernels = self.get_gabor_kernels(kernel_size, sigma, lambd, gamma, psi, theta_values, image.device)
        responses = []
        for k in kernels:
            k_expanded = k.expand(C, 1, -1, -1)
            filtered = F.conv2d(image, k_expanded, padding=kernel_size // 2, groups=C)
            responses.append(filtered.squeeze(0))
        return torch.stack(responses, dim=0).mean(dim=0)

    def get_gabor_kernels(self, kernel_size, sigma, lambd, gamma, psi, theta_values, device):
        half = kernel_size // 2
        y, x = torch.meshgrid(torch.linspace(-half, half, kernel_size, device=device), torch.linspace(-half, half, kernel_size, device=device), indexing='ij')
        kernels = []
        for theta in theta_values:
            x_theta = x * torch.cos(theta) + y * torch.sin(theta)
            y_theta = -x * torch.sin(theta) + y * torch.cos(theta)
            gb = torch.exp(-0.5 * (x_theta**2 + (gamma**2) * y_theta**2) / sigma**2) * torch.cos(2 * math.pi * x_theta / lambd + psi)
            kernels.append(gb)
        return torch.stack(kernels).unsqueeze(1)

    def face_restorer_auto(self, original_face_512, swap_original, swap, alpha, adjust_sharpness, scale_factor, swap_mask, kernel=0, pixelwise=False):
        original_face_512_autorestore = original_face_512.clone().float()
        original_face_512 = original_face_512 * swap_mask
        swap_autorestore = swap.clone()
        swap = swap * swap_mask
        swap_original_autorestore = swap_original.clone()
        swap_original = swap_original * swap_mask

        if pixelwise:
            alpha_range = adjust_sharpness/100
            swap = swap_autorestore * max(0.0, alpha) + swap_original_autorestore * min(1.0, 1 - alpha)
            swap = swap * swap_mask
            sharp_orig_map = self.tenengrad_sharpness_map(original_face_512)
            sharp_swap_map = self.tenengrad_sharpness_map(swap)
            diff_map = sharp_orig_map - sharp_swap_map
            
            pos_values = diff_map[diff_map > 0]
            pos_thresh = torch.quantile(pos_values, 0.95) if pos_values.numel() > 0 else torch.tensor(0.0, device=diff_map.device)
            diff_map = torch.clamp(diff_map, min=0, max=pos_thresh)
            diff_min, diff_max = diff_map.min(), diff_map.max()
            diff_map_norm = (diff_map - diff_min) / (diff_max - diff_min + 1e-8)            
            alpha_map = diff_map_norm * alpha_range
            
            if torch.isnan(alpha_map).any():
                return original_face_512_autorestore, alpha_map
            else:
                alpha_map = transforms.GaussianBlur(kernel*2+1, (kernel+1)*0.2)(alpha_map.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
                final_alpha_map = (alpha + alpha_map).clamp(0, 1)
                alpha_map_rgb = final_alpha_map.unsqueeze(0).repeat(3, 1, 1)
                blended = swap_autorestore * alpha_map_rgb + swap_original_autorestore * (1 - alpha_map_rgb)
                return blended, alpha_map

        sharpness_original = self.tenengrad_sharpness(original_face_512)
        max_iterations, tolerance, min_alpha_change = 7, 5.0, 0.05
        alpha_min, alpha_max = 0.0, 1.0
        iteration, iteration_blur = 0, 0
        prev_alpha = alpha

        while iteration < max_iterations:
            swap2 = swap * alpha + swap_original * (1 - alpha)
            swap2_masked = torch.where(original_face_512 != 0.0, swap2, torch.tensor(0.0, device=swap2.device))
            sharpness_swap = self.tenengrad_sharpness(swap2_masked)
            sharpness_diff = sharpness_swap - sharpness_original
            if abs(sharpness_diff) < tolerance: break
            if sharpness_diff < 0: alpha_min, alpha = alpha, (alpha + alpha_max) / 2
            else: alpha_max, alpha = alpha, (alpha + alpha_min) / 2
            if alpha < 0.07:
                swap_blur = swap_original.clone()
                prev_alpha = 0
                for i_blur in range(7):
                    iteration_blur = i_blur
                    swap2 = transforms.GaussianBlur(2 * i_blur + 1, i_blur * 0.2)(swap_blur) if i_blur != 0 else swap_blur
                    swap2_masked = torch.where(original_face_512_autorestore != 0.0, swap2, torch.tensor(0.0, device=swap2.device))
                    if self.tenengrad_sharpness(swap2_masked) - sharpness_original <= 0: break
                break
            if abs(prev_alpha - alpha) < min_alpha_change: break
            prev_alpha = alpha
            iteration += 1
        return prev_alpha, iteration_blur
        
    def tenengrad_sharpness(self, image):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gray_image = torch.mean(image.float(), dim=0, keepdim=True)
        grad_x = F.conv2d(gray_image.unsqueeze(0), sobel_x, padding=1)
        grad_y = F.conv2d(gray_image.unsqueeze(0), sobel_y, padding=1)
        gradient_energy = torch.mean(grad_x**2 + grad_y**2)
        gradient_energy_2d = gradient_energy.squeeze(0).squeeze(0)
        gray_image_2d = gray_image.squeeze(0)
        valid_mask = (gray_image_2d != 0.0).float()
        valid_count = valid_mask.sum()
        if valid_count.item() == 0: return torch.tensor(0.0, device=image.device)
        return (gradient_energy_2d * valid_mask).sum() / valid_count
        
    def tenengrad_sharpness_map(self, image):
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=image.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=image.device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gray = torch.mean(image.float(), dim=0, keepdim=True).unsqueeze(0)
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        return (grad_x ** 2 + grad_y ** 2).squeeze(0).squeeze(0)
        
    def apply_block_shift_gpu(self, img, block_size=8, shift_max=2):
        block_size = 2 ** block_size
        C, H, W = img.shape
        img = img.float()
        H_crop, W_crop = H - (H % block_size), W - (W % block_size)
        img = img[:, :H_crop, :W_crop]
        H_blocks, W_blocks = H_crop // block_size, W_crop // block_size
        shift_x = torch.randint(-shift_max, shift_max + 1, (H_blocks, W_blocks), device=img.device)
        shift_y = torch.randint(-shift_max, shift_max + 1, (H_blocks, W_blocks), device=img.device)
        base_grid = F.affine_grid(torch.eye(2, 3, device=img.device).unsqueeze(0), [1, C, H_crop, W_crop], align_corners=False)
        shift_x = shift_x.float() * (2 / W_crop)
        shift_y = shift_y.float() * (2 / H_crop)
        shift_x = shift_x.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        shift_y = shift_y.repeat_interleave(block_size, dim=0).repeat_interleave(block_size, dim=1)
        base_grid[..., 0] += shift_x
        base_grid[..., 1] += shift_y
        distorted_img = F.grid_sample(img.unsqueeze(0), base_grid, mode='bilinear', padding_mode='border', align_corners=False)
        return distorted_img.squeeze(0).clamp(0, 255)
        
    def analyze_image(self, image):
        image = image.float() / 255.0
        grayscale = torch.mean(image, dim=0, keepdim=True)
        analysis = {}
        fft = torch.fft.fft2(grayscale)
        high_freq_energy = torch.mean(torch.abs(fft))
        analysis["jpeg_artifacts"] = min(high_freq_energy.item() / 50, 1.0)
        median_filtered = F.avg_pool2d(grayscale, 3, stride=1, padding=1)
        noise_map = torch.abs(grayscale - median_filtered)
        sp_noise = torch.mean((noise_map > 0.1).float())
        analysis["salt_pepper_noise"] = min(sp_noise.item() * 10, 1.0)
        local_var = F.avg_pool2d(grayscale**2, 5, stride=1, padding=2) - (F.avg_pool2d(grayscale, 5, stride=1, padding=2) ** 2)
        speckle_noise = torch.mean(local_var)
        analysis["speckle_noise"] = min(speckle_noise.item() * 50, 1.0)
        laplace_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=image.device).unsqueeze(0).unsqueeze(0)
        laplace_edges = F.conv2d(grayscale.unsqueeze(0), laplace_kernel, padding=1)
        edge_strength = torch.mean(torch.abs(laplace_edges))
        analysis["blur"] = 1.0 - min(edge_strength.item() * 5, 1.0)
        contrast = grayscale.std()
        analysis["low_contrast"] = 1.0 - min(contrast.item() * 10, 1.0)
        return analysis