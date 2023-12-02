from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np


class PbModel:
    def __init__(self, input_saved_model_file_path: str = None, classes: Tuple = None):
        self.model = tf.keras.models.load_model(input_saved_model_file_path)
        self.grad_model = self.__prepare_grad_model()
        self.model_input_shape = self.model.inputs[0].shape
        self.model_input_dtype = self.model.inputs[0].dtype
        self.model_output_count_shape = self.model.outputs[-2].shape
        self.model_output_count_dtype = self.model.outputs[-2].dtype
        self.model_output_cam_shape = self.model.outputs[-1].shape
        self.model_output_cam_dtype = self.model.outputs[-1].dtype
        self.classes = classes

    def inference(self, input_video_list: List[np.ndarray]) -> Tuple[List[Dict], Tuple]:
        resized_video_array_list, resize_video_shape_list, input_video_shape_list = self.__preprocess_video_list(input_video_list, self.model_input_shape[2:4])
        output_count_tensor_list, output_cam_tensor_list, output_grad_cam_tensor_list = self.__inference(resized_video_array_list)
        output = self.__output_parse(output_count_tensor_list, output_cam_tensor_list, output_grad_cam_tensor_list, resize_video_shape_list, input_video_shape_list)
        return output, (output_count_tensor_list, output_cam_tensor_list, output_grad_cam_tensor_list)

    def __inference(self, resized_video_array_list: List[np.ndarray]) -> np.ndarray:
        output_count_tensor_list, output_cam_tensor_list = [], []
        for resized_video_array in resized_video_array_list:
            batch = tf.convert_to_tensor(np.expand_dims(resized_video_array, axis=0), dtype=self.model_input_dtype)
            raw_count_pred, raw_cam_pred = self.model(batch)
            output_count_tensor_list.append(raw_count_pred.numpy())
            output_cam_tensor_list.append(raw_cam_pred.numpy())
            output_grad_cam_tensor_list = [tf.zeros((batch.shape[0], raw_cam_pred.shape[1], output.shape[2], output.shape[3], self.model_output_count_shape[-1]),
                                                    self.model_output_count_dtype).numpy() for output in self.grad_model.outputs[:-2]]
            for pred_index in range(self.model_output_count_shape[-1]):
                raw_grad_list = self.__make_gradcam_heatmap(batch, pred_index)
                for grad_index, raw_grad in enumerate(raw_grad_list):
                    output_grad_cam_tensor_list[grad_index][0, :, :, :, pred_index] = raw_grad
        return output_count_tensor_list, output_cam_tensor_list, output_grad_cam_tensor_list

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

    def __output_parse(self, output_count_tensor_list, output_cam_tensor_list, output_grad_cam_tensor_list, resize_video_shape_list, input_video_shape_list) -> List[Dict]:
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
                        resize_crop_cam_video_sum = np.sum(resize_crop_cam_video)
                        if resize_crop_cam_video_sum != 0.0:
                            resize_crop_cam_video *= (org_cam_sum / np.sum(resize_crop_cam_video))
                        elif org_cam_sum != 0.0 and resize_crop_cam_video_sum == 0.0:
                            resize_crop_cam_video[0][0] = org_cam_sum
                        resize_cam_canvas[frame_index, :, :, cam_index] = self.__make_count_heatmap(
                            output_cam_tensor[0, frame_index, :, :, cam_index],
                            resize_video_shape_list[index][0], resize_video_shape_list[index][1],
                            input_video_shape_list[index][1], input_video_shape_list[index][2],output_count_tensor[0][cam_index])
                resize_grad_cam_canvas_list = []
                for grad_index in range(len(output_grad_cam_tensor_list)):
                    resize_grad_cam_canvas =   np.zeros((input_video_shape_list[index][0], input_video_shape_list[index][1], input_video_shape_list[index][2], output_count_tensor.shape[-1]), dtype=np.float32)
                    for frame_index in range(input_video_shape_list[index][0]):
                        for pred_index in range(output_cam_tensor.shape[-1]):
                            resize_grad_cam_canvas[frame_index, :, :, pred_index] = self.__make_count_heatmap(
                                output_grad_cam_tensor_list[grad_index][0, frame_index, :, :, pred_index],
                                resize_video_shape_list[index][0], resize_video_shape_list[index][1],
                                input_video_shape_list[index][1], input_video_shape_list[index][2],output_count_tensor[0][cam_index])
                    for pred_index in range(output_cam_tensor.shape[-1]):
                        if np.all(resize_grad_cam_canvas[:, :, :, pred_index]  == 0.0):
                            resize_grad_cam_canvas[:, :, :, pred_index] = np.zeros(resize_grad_cam_canvas[:, :, :, pred_index].shape, dtype=np.float32)
                        else:
                            resize_grad_cam_canvas[:, :, :, pred_index]  = resize_grad_cam_canvas[:, :, :, pred_index] * (output_count_tensor[index][pred_index]/np.sum(resize_grad_cam_canvas[:, :, :, pred_index]))
                    resize_grad_cam_canvas_list.append(resize_grad_cam_canvas)
                output_dict = {'count': [count for count in output_count_tensor[0].tolist()],
                               'cam': resize_cam_canvas,
                               'grad_cam': resize_grad_cam_canvas_list}
            output_dict_list.append(output_dict)
        return output_dict_list

    def __prepare_grad_model(self, layer_name_list=('conv3d_0', 'conv3d_1', 'conv3d_2', 'conv3d_3')):
        output_layers = []
        for layer_name in layer_name_list:
            output_layers.append(self.model.get_layer(layer_name).output)
        output_layers.append(self.model.output)
        grad_model = tf.keras.models.Model([self.model.inputs], output_layers)
        return grad_model

    def __make_gradcam_heatmap(self, video, pred_index):
        with tf.GradientTape(persistent=True) as tape:
            outputs = self.grad_model(video)
            conv_outputs_list = outputs[:-1]
            predictions = outputs[-1][0]
            loss = predictions[:, pred_index]
        cam_list = []
        for conv_outputs in conv_outputs_list:
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
            cam_list.append(heatmap.numpy())
        del tape
        return cam_list

    def __make_count_heatmap(self, image_array, crop_height, crop_width, resize_height, resize_width, total_num=None):
        if np.all(image_array == 0.0):
            return np.zeros((resize_height, resize_width), dtype=np.float32)
        org_input_image = Image.fromarray(image_array).resize((self.model_input_shape[3], self.model_input_shape[2]), resample=Image.BICUBIC)
        crop_image = org_input_image.crop((0, 0, crop_width, crop_height))
        resize_crop_image = np.asarray(crop_image.resize((resize_width, resize_height)))
        resize_crop_image = resize_crop_image - np.min(resize_crop_image)
        resize_crop_image = np.clip(resize_crop_image, 0, np.max(resize_crop_image))
        if np.all(resize_crop_image == 0.0):
            return np.zeros((resize_height, resize_width), dtype=np.float32)
        if total_num is not None:
            resize_crop_image = resize_crop_image * (total_num/np.sum(resize_crop_image))
        return resize_crop_image