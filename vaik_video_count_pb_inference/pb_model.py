from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.model_input_shape = self.model.signatures["serving_default"].inputs[0].shape
        self.model_input_dtype = self.model.signatures["serving_default"].inputs[0].dtype
        self.model_output_count_shape = self.model.signatures["serving_default"].outputs[0].shape
        self.model_output_count_dtype = self.model.signatures["serving_default"].outputs[0].dtype
        self.model_output_cam_shape = self.model.signatures["serving_default"].outputs[1].shape
        self.model_output_cam_dtype = self.model.signatures["serving_default"].outputs[1].dtype
        self.classes = classes

    def inference(self, input_video_list: List[np.ndarray]) -> Tuple[List[Dict], Tuple]:
        resized_video_array_list, resize_video_shape_list, input_video_shape_list = self.__preprocess_video_list(input_video_list, self.model_input_shape[2:4])
        output_count_tensor_list, output_cam_tensor_list = self.__inference(resized_video_array_list)
        output = self.__output_parse(output_count_tensor_list, output_cam_tensor_list, resize_video_shape_list, input_video_shape_list)
        return output, (output_count_tensor_list, output_cam_tensor_list)

    def __inference(self, resized_video_array_list: List[np.ndarray]) -> np.ndarray:
        output_count_tensor_list, output_cam_tensor_list = [], []
        for resized_video_array in resized_video_array_list:
            batch = tf.convert_to_tensor(np.expand_dims(resized_video_array, axis=0), dtype=self.model_input_dtype)
            raw_count_pred, raw_cam_pred = self.model(batch)
            output_count_tensor_list.append(raw_count_pred.numpy())
            output_cam_tensor_list.append(raw_cam_pred.numpy())
        return output_count_tensor_list, output_cam_tensor_list

    def __preprocess_video_list(self, input_video_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_video_list, resize_video_shape_list, input_video_shape_list = [], [], []
        
        for input_video in input_video_list:
            resized_video, resize_video_shape, input_video_shape = self.__preprocess_video(input_video, resize_input_shape)
            resized_video_list.append(resized_video)
            resize_video_shape_list.append(resize_video_shape)
            input_video_shape_list.append(input_video_shape)
        return resized_video_list, resize_video_shape_list, input_video_shape_list

    def __preprocess_video(self, input_video: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_video.shape) != 4:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_video.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_video.dtype}')

        output_video = np.zeros((input_video.shape[0], *resize_input_shape, input_video.shape[-1]),
                                dtype=input_video.dtype)
        for frame_index, input_image in enumerate(input_video):
            pil_image = Image.fromarray(input_image)
            x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
            if x_ratio < y_ratio:
                resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
            else:
                resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
            resize_pil_image = pil_image.resize(resize_size)
            resize_image = np.array(resize_pil_image)
            output_video[frame_index, :resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_video, resize_image.shape, input_video.shape

    def __output_parse(self, output_count_tensor_list, output_cam_tensor_list, resize_video_shape_list, input_video_shape_list) -> List[Dict]:
        output_dict_list = []
        for output_count_tensor, output_cam_tensor in zip(output_count_tensor_list, output_cam_tensor_list):
            for index in range(output_count_tensor.shape[0]):
                resize_cam_canvas = np.zeros(input_video_shape_list[index], dtype=np.float32)
                for frame_index in range(output_cam_tensor.shape[1]):
                    for cam_index in range(output_cam_tensor.shape[-1]):
                        org_cam_sum = np.sum(output_cam_tensor[0, frame_index, :, :, cam_index])
                        cam_input_video = Image.fromarray(output_cam_tensor[0, frame_index, :, cam_index]).resize((self.model_input_shape[2], self.model_input_shape[3]), resample=Image.BICUBIC)
                        crop_cam_video = cam_input_video.crop((0, 0, resize_video_shape_list[index][1], resize_video_shape_list[index][0]))
                        resize_crop_cam_video = np.asarray(crop_cam_video.resize((input_video_shape_list[index][2], input_video_shape_list[index][1])))
                        resize_crop_cam_video = np.clip(resize_crop_cam_video, 0, np.max(resize_crop_cam_video))
                        if org_cam_sum != 0:
                            resize_crop_cam_video *= org_cam_sum / np.sum(resize_crop_cam_video)
                        resize_cam_canvas[frame_index, :, :, cam_index] = resize_crop_cam_video
                output_dict = {'count': [count for count in output_count_tensor[0].tolist()],
                               'cam': resize_cam_canvas}
            output_dict_list.append(output_dict)
        return output_dict_list