import torchvision.transforms as transforms

import nvidia.dali.plugin.pytorch as dalitorch
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy, DALIClassificationIterator
from nvidia.dali import Pipeline, pipeline_def, fn, types
import os

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

#def create_pipeline_compcars(batch_size, num_threads, data_dir, input_size):

class PipelineCompCars:
    def __init__(self, batch_size, num_threads, data_dir, input_size, prefetch=1):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.data_dir = data_dir
        self.input_size = input_size
        self.prefetch = prefetch
        self.make_pipe()
        self.pipe.build()
        self.dataset = DALIGenericIterator(
            self.pipe,
            ["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def make_pipe(self):
        self.pipe = Pipeline(
            batch_size=self.batch_size, 
            num_threads=self.num_threads, 
            device_id=0,
            prefetch_queue_depth=self.prefetch
        )
        with self.pipe:
            inputs, labels = fn.readers.file(
                file_root=self.data_dir,
                random_shuffle=True,
                prefetch_queue_depth=self.prefetch,
                name="Reader",
            )
            output_layout = "CHW"
            
            images = fn.decoders.image(inputs, device = "mixed")

            images = fn.resize(
                images,
                dtype=types.UINT8,
                resize_x=self.input_size,
                resize_y=self.input_size,
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
            self.pipe.set_outputs(images, labels)

class PipelineImageNet:
    def __init__(self, batch_size, num_threads, data_dir, input_size, prefetch=2):
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.data_dir = data_dir
        self.input_size = input_size
        self.prefetch = prefetch
        self.make_pipe()
        self.pipe.build()
        self.dataset = DALIClassificationIterator(
            self.pipe,
            #["data", "label"],
            reader_name="Reader",
            last_batch_policy=LastBatchPolicy.DROP,
        )

    def make_pipe(self):
        self.pipe = Pipeline(
            batch_size=self.batch_size, 
            num_threads=self.num_threads, 
            device_id=0,
            prefetch_queue_depth=self.prefetch
        )
        #device_memory_padding = 211025920
        #host_memory_padding = 140544512
        # ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
        #preallocate_width_hint = 5980
        #preallocate_height_hint = 6430
        with self.pipe:
            images, labels = fn.readers.file(file_root=self.data_dir,
                                     random_shuffle=True,
                                     pad_last_batch=True,
                                     name="Reader")
            
            images = fn.decoders.image_random_crop(images,
                                               device="mixed", output_type=types.RGB,
                                               #device_memory_padding=device_memory_padding,
                                               #host_memory_padding=host_memory_padding,
                                               #preallocate_width_hint=preallocate_width_hint,
                                               #preallocate_height_hint=preallocate_height_hint,
                                               random_aspect_ratio=[0.8, 1.25],
                                               random_area=[0.1, 1.0],
                                               num_attempts=100)
            images = fn.resize(images,
                            device="gpu",
                            resize_x=self.input_size,
                            resize_y=self.input_size,
                            interp_type=types.INTERP_TRIANGULAR)
            mirror = fn.random.coin_flip(probability=0.5)

            images = fn.crop_mirror_normalize(images.gpu(),
                                      dtype=types.FLOAT,
                                      output_layout="CHW",
                                      crop=(self.input_size, self.input_size),
                                      mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                      std=[0.229 * 255,0.224 * 255,0.225 * 255],
                                      mirror=mirror)

            labels = fn.squeeze(labels, axes=[0])
            labels = labels.gpu()
            self.pipe.set_outputs(images, labels)

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


class DALIDataset:
    def __init__(self, dataset_name, path, batch_size, num_workers, input_size):
        self.path = path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_size = input_size

        if dataset_name == "compcars":
            pipe_class = PipelineCompCars(
                self.batch_size, 
                self.num_workers, 
                self.path, 
                self.batch_size
            )
            self.dataset = pipe_class.dataset

        if dataset_name in ("imagenet", "imagenet64x64", "imagenet64_images"):
            pipe_class = PipelineImageNet(
                self.batch_size, 
                self.num_workers, 
                self.path, 
                self.batch_size
            )
            self.dataset = pipe_class.dataset

    def __len__(self):
        return sum([len(files) for r, d, files in os.walk(self.path)])