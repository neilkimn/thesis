import torchvision.transforms as transforms

import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali import Pipeline, pipeline_def, fn, types

INPUT_SIZE = 224

random_perspective_transform = transforms.Compose([
                              transforms.RandomPerspective(p=0.5),
                              ])
def perspective(t):
    return random_perspective_transform(t)

def mux(condition, true_case, false_case):
    neg_condition = condition ^ True
    return condition * true_case + neg_condition * false_case

def create_pipeline_perspective(batch_size, num_threads, data_dir):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads, exec_async=False, exec_pipelined=False)
    def _create_pipeline(data_dir):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            prefetch_queue_depth=2,
            name="Reader",
        )

        output_layout = "CHW"
        
        images = fn.decoders.image(inputs, device = "cpu")

        images = fn.resize(
            images,
            dtype=types.UINT8,
            resize_x=INPUT_SIZE,
            resize_y=INPUT_SIZE,
            device="cpu"
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, vertical = flip_coin, device = "cpu") # Vertical Flip

        contrast_jitter = fn.random.uniform(range=[0.5, 1.5])
        saturation_jitter = fn.random.uniform(range=[0.5, 1.5])
        hue_jitter = fn.random.uniform(range=[0.5, 1.5])

        images_jittered = fn.color_twist(
            images, 
            contrast=contrast_jitter, 
            saturation=saturation_jitter,
            hue=hue_jitter
        )
        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images = mux(condition, images_jittered, images) # Color Jitter

        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images_gray = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
        images = mux(condition, images_gray, images) # Gray Scale

        images = fn.transpose(images, output_layout=output_layout) # Transform from HWC -> CHW

        images = dalitorch.fn.torch_python_function(images, function=perspective) # Random Perspective

        images = fn.normalize(images, dtype=types.FLOAT)

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir)

def create_pipeline_no_perspective(batch_size, num_threads, data_dir, input_size):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads)
    def _create_pipeline(data_dir, input_size):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            prefetch_queue_depth=1,
            name="Reader",
        )

        output_layout = "CHW"
        
        images = fn.decoders.image(inputs, device = "mixed")

        images = fn.resize(
            images,
            dtype=types.UINT8,
            resize_x=input_size,
            resize_y=input_size,
            device="gpu"
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, vertical = flip_coin, device = "gpu") # Vertical Flip

        contrast_jitter = fn.random.uniform(range=[0.5, 1.5])
        saturation_jitter = fn.random.uniform(range=[0.5, 1.5])
        hue_jitter = fn.random.uniform(range=[0.5, 1.5])

        images_jittered = fn.color_twist(
            images, 
            contrast=contrast_jitter, 
            saturation=saturation_jitter,
            hue=hue_jitter
        )
        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images = mux(condition, images_jittered, images) # Color Jitter

        condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL)
        images_gray = fn.color_space_conversion(images, image_type=types.RGB, output_type=types.GRAY)
        images = mux(condition, images_gray, images) # Gray Scale

        images = fn.transpose(images, output_layout=output_layout) # Transform from HWC -> CHW

        images = fn.normalize(images, dtype=types.FLOAT)

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir, input_size)

def create_pipeline_imagenet(batch_size, num_threads, data_dir, input_size):

    @pipeline_def(device_id=0, batch_size=batch_size, num_threads=num_threads)
    def _create_pipeline(data_dir, input_size):
        inputs, labels = fn.readers.file(
            file_root=data_dir,
            random_shuffle=True,
            prefetch_queue_depth=1,
            name="Reader",
        )

        output_layout = "CHW"
        
        images = fn.decoders.image(inputs, device = "mixed")


        images = fn.random_resized_crop(
            images,
            size=input_size,
            dtype=types.UINT8, 
            device="gpu"
        )

        flip_coin = fn.random.coin_flip(probability=0.5)
        images = fn.flip(images, horizontal = flip_coin, device = "gpu") # Horizontal Flip

        
        images = fn.transpose(images, output_layout=output_layout) # Transform from HWC -> CHW

        images = fn.normalize(
            images, 
            dtype=types.FLOAT,
            #mean=[0.485, 0.456, 0.406],
            #stddev=[0.229, 0.224, 0.225]
            )

        labels = fn.squeeze(labels, axes=[0])

        return images.gpu(), labels.gpu()
    return _create_pipeline(data_dir, input_size)



from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy
import os

class DALIDataset:
    def __init__(self, dataset_name, path, batch_size, num_workers, input_size):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        if dataset_name == "compcars_dali":
            pipeline = create_pipeline_no_perspective(
                self.batch_size, 
                self.num_workers, 
                self.path,
                self.input_size,
            )
        if dataset_name[:8] == "imagenet":
            pipeline = create_pipeline_imagenet(
                self.batch_size,
                self.num_workers,
                self.path,
                self.input_size,
            )
        self.dataset = DALIGenericIterator(
            pipeline,
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.PARTIAL,
        )
    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.path)])