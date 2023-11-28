# vaik-video-count-pb-inference

Inference by count for video PB model

## Install

```shell
pip install git+https://github.com/vaik-info/vaik-video-count-pb-inference.git
```


## Usage
### Example

```python
import os

import imageio
import numpy as np

from vaik_video_count_pb_inference.pb_model import PbModel

input_saved_model_dir_path = os.path.expanduser('~/.vaik-video-count-pb-trainer/output_model/2023-11-28-07-10-44/step-1000_batch-8_epoch-34_loss_0.1202_val_loss_0.1023')

video_path = os.path.expanduser('/media/kentaro/dataset/.vaik-mnist-video-count-dataset/valid/valid_000000000.avi')
classes = ('zero', 'one', 'two')
video = imageio.get_reader(video_path,  'ffmpeg')
frames = np.stack([np.array(frame, dtype=np.uint8) for frame in video], axis=0)

model = PbModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([frames])
```

## Output

- output
```
[{'count': [4.8042, 2.8184, 0.9538], 'cam': array([[[[0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        [0.0000000e+00, 0.0000000e+00, 0.0000000e+00],
        ・・・], dtype=float32)},
        ・・・]
```

- raw_pred

```
(array([[0.9881473 , 0.        , 1.9089862 ],
       ・・・], dtype=float32), array([[[[[0.        , 0.        , 0.        ],
         [0.        , 0.        , 0.1        ],
        ...,], dtype=float32))
```
