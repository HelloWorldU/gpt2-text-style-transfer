
# LuoBiFengYu - Text Style Transfer Model

This project is aimed at achieving text style transfer by training generator and discriminator models. The generator model is fine-tuned based on GPT-2, and the discriminator models are built using BERT and a simple feedforward neural network.

## Project Structure

```
├── model
│   ├── generator
│   ├── discriminator_Y
│   └── discriminator_Z
├── train_model.py
├── requirements.txt
└── README.md
```

- `model/` directory contains the trained model parameters.
- `train_model.py` is the main script for model training.
- `requirements.txt` lists the dependencies required for the project.

## Installation

Make sure you have [Python 3.8](https://www.python.org/downloads/) or higher installed. It is recommended to use [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to create a virtual environment.

1. Clone the project repository:

   ```bash
   git clone https://github.com/your_username/your_repository_name.git
   cd your_repository_name
   ```

2. Create and activate the virtual environment:

   ```bash
   conda create -n gpt2-env python=3.8
   conda activate gpt2-env
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Train the model:

   Run the following command in the project root directory to start training the model:

   ```bash
   python train_model.py
   ```

   After training, the model parameters will be saved in the `model/` directory.

2. Generate text:

   You can generate stylized text by invoking the generator model:

   ```python
   from transformers import GPT2Tokenizer, TFGPT2LMHeadModel

   tokenizer = GPT2Tokenizer.from_pretrained('./model/generator')
   model = TFGPT2LMHeadModel.from_pretrained('./model/generator')

   input_text = "Your input text"
   input_ids = tokenizer.encode(input_text, return_tensors='tf')

   output = model.generate(input_ids)
   generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

   print(generated_text)
   ```

## Mixed Precision Training

This project uses mixed precision training to improve efficiency. If you encounter issues with type mismatches between `float16` and `float32`, ensure that the data types are consistent:

```python
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')
```

## Dependencies

- TensorFlow >= 2.8
- transformers >= 4.43.3
- CUDA >= 11.2
- cuDNN >= 8.8

For a detailed list of dependencies, please refer to the `requirements.txt` file.

## Contributing

Feel free to submit issues and feature requests! If you want to contribute code, please fork this repository, create a new branch, and submit your changes via a Pull Request.

## License

This project is licensed under the Apache2.0 License. See the [LICENSE](./LICENSE) file for details.
