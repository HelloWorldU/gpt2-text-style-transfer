import tensorflow as tf
import os
import re
import matplotlib.pyplot as plt
import random 

i=1
# 绘制困惑度曲线
def perplexity_curve(perplexities, save_dir):
    global i
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(perplexities) + 1), perplexities, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Perplexity Curve')
    plt.savefig(os.path.join(save_dir,f'perplexity_curve{i+1}.png'))
    i+=1
    plt.close()

# 绘制损失值曲线
def loss_curve(train_loss, valid_loss, test_loss, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_loss) + 1), train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, marker='o', label='Valid Loss')
    plt.plot(range(1, len(test_loss) + 1), test_loss, marker='o', label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'loss_curve{i}.png'))
    plt.close()

# 绘制准确率曲线
def accuracy_curve(train_accuracy, valid_accuracy, test_accuracy, save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, marker='o', label='Train Accuracy')
    plt.plot(range(1, len(valid_accuracy) + 1), valid_accuracy, marker='o', label='Valid Accuracy')
    plt.plot(range(1, len(test_accuracy) + 1), test_accuracy, marker='o', label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'accuracy_curve{i}.png'))
    plt.close()

# 绘制学习率曲线
def learning_rate_curve(learning_rates,save_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(learning_rates) + 1),learning_rates,marker='o',label='Learning Rate')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir,f'learning_rate_curve{i}.png'))
    plt.close()

# 自定义学习率调度器
class WarmUpDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, decay_schedule_fn, warmup_steps):
        super(WarmUpDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_schedule_fn = decay_schedule_fn
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        def warmup():
            return self.initial_learning_rate * (step / self.warmup_steps)
        def decay():
            return self.decay_schedule_fn(step - self.warmup_steps)
        
        # 图模式下使用 tf.cond 进行条件判断
        return tf.cond(step < self.warmup_steps, warmup, decay)
    
# 生成文本并保存到文件
def generate_text(model, tokenizer, prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors="tf")
    outputs = model.generate(**inputs, max_length=max_length, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 删除中文字符之间的空格
    generated_text = re.sub(r'([\u4e00-\u9fff])\s+([\u4e00-\u9fff])', r'\1\2', generated_text)
    
    # 删除标点符号前后的空格
    generated_text = re.sub(r'\s+([。，！？：；\'\"）】》])', r'\1', generated_text)
    generated_text = re.sub(r'([\'\"（【《])\s+', r'\1', generated_text)
    
    # 删除多余的空格，但保留英文单词之间的单个空格
    generated_text = re.sub(r'\s+', ' ', generated_text).strip()
    
    # 删除非中文、英文、数字和指定标点之外的字符
    generated_text = re.sub(r'[^\w\u4e00-\u9fff。，！？""''：……]', '', generated_text)
    
    return generated_text

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 读取数据集
def load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, split_ratio=0.9, seed=None):
    if seed is not None:
        random.seed(seed)

    def read_and_tokenize(file_path, style):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.read().splitlines()
        random.shuffle(lines)
        split = int(len(lines) * split_ratio)
        train_lines = lines[:split]
        valid_lines = lines[split:]
        
        train_encoded = tokenizer(train_lines, truncation=True, padding=False)
        valid_encoded = tokenizer(valid_lines, truncation=True, padding=False)

        train_dataset = (
            train_encoded['input_ids'],
            train_encoded['attention_mask'],
            [style] * len(train_lines)
        )
        valid_dataset = (
            valid_encoded['input_ids'],
            valid_encoded['attention_mask'],
            [style] * len(valid_lines)
        )
        
        return train_dataset, valid_dataset

    train_dataset_X, valid_dataset_X = read_and_tokenize(file_path_X, 0)
    train_dataset_Y, valid_dataset_Y = read_and_tokenize(file_path_Y, 1)

    with open(test_file_path, 'r', encoding='utf-8') as f_test:
        test_lines = f_test.read().splitlines()
    test_encoded = tokenizer(test_lines, truncation=True, padding=False)
    test_dataset = (
        test_encoded['input_ids'], 
        test_encoded['attention_mask'], 
        [0] * len(test_lines)
    )

    return (train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, test_dataset)

def create_labels(input_ids, attention_mask):
    max_len = max(len(seq) for seq in input_ids)
    # 确保 input_ids 和 attention_mask 是 Python 列表
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=max_len, padding='post')
    attention_mask = tf.keras.preprocessing.sequence.pad_sequences(attention_mask, maxlen=max_len, padding='post')
    
    labels = tf.roll(input_ids, shift=-1, axis=1)
    labels = tf.where(attention_mask == 0, -100, labels)
    return labels

# 创建数据集
def create_tf_dataset(dataset, batch_size, drop_remainder=True):
    input_ids, attention_mask, styles = dataset
    
    def gen():
        for i in range(len(input_ids)):
            yield {'input_ids': input_ids[i], 
                   'attention_mask': attention_mask[i], 
                   'style': styles[i]}
    
    tf_dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'style': tf.TensorSpec(shape=(), dtype=tf.int32)
        }
    )
    
    padded_shapes = {
        'input_ids': tf.TensorShape([None]),
        'attention_mask': tf.TensorShape([None]),
        'style': tf.TensorShape([])
    }
    
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=padded_shapes, drop_remainder=drop_remainder)
    
    return tf_dataset

# 测试集评估函数
def test_evalution(generator, test_step,dataset):
    test_perplexity = 0.0
    total_test_loss = tf.constant(0.0,dtype=tf.float32)
    total_test_accuracy = tf.constant(0.0,dtype=tf.float32)
    num_test_batches = 0
    for batch_test_X in dataset:
        batch_input_ids = batch_test_X['input_ids']
        batch_attention_mask = batch_test_X['attention_mask']
        batch_labels = create_labels(batch_input_ids, batch_attention_mask)
        styles=batch_test_X['style']
        batch_attention_mask = tf.convert_to_tensor(batch_attention_mask, tf.int32)
        loss,accuracy = test_step(generator, batch_input_ids, batch_attention_mask, batch_labels, styles)
        total_test_loss += loss
        total_test_accuracy += accuracy
        num_test_batches += 1
    avg_test_loss = total_test_loss / tf.cast(num_test_batches, tf.float32)
    avg_test_accuracy = total_test_accuracy / tf.cast(num_test_batches, tf.float32)
    test_perplexity = tf.exp(avg_test_loss).numpy()
    print(f"Test Loss: {avg_test_loss}, Test Accuracy: {avg_test_accuracy}")
    print(f"Test Perplexity: {test_perplexity}")
    return avg_test_accuracy,avg_test_loss,test_perplexity

# 绘制生成器鉴别器的各个损失
def plot_losses(rec_losses, lm_losses, adv_losses, kl_losses, disc_losses, disc_z_losses, save_dir):
    plt.figure(figsize=(12, 8))
    plt.plot(rec_losses, label='Reconstruction Loss')
    plt.plot(lm_losses, label='Language Model Loss')
    plt.plot(adv_losses, label='Adversarial Loss')
    plt.plot(kl_losses, label='KL Divergence Loss')
    plt.plot(disc_losses, label='Discriminator Loss')
    plt.plot(disc_z_losses, label='Discriminator Z Loss')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'losses.png'))
    plt.close()

# 验证和测试步骤
@tf.function
def valid_step(generator, input_ids, attention_mask, labels, styles):
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    styles = tf.convert_to_tensor(styles, dtype=tf.int32)

    outputs = generator(input_ids=input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, tf.float32)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(tf.cast(loss, tf.float32) * mask) / tf.reduce_sum(mask)
    predictions=tf.argmax(logits, axis=-1)
    predictions=tf.cast(predictions,tf.int32)
    accuracy=tf.reduce_sum(tf.cast(tf.equal(predictions,labels),tf.float32)*mask)/tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32),accuracy

@tf.function
def test_step(generator, input_ids, attention_mask, labels, styles):
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    styles = tf.convert_to_tensor(styles, dtype=tf.int32)

    outputs = generator(input_ids=input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, tf.float32)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(tf.cast(loss, tf.float32) * mask) / tf.reduce_sum(mask)
    predictions = tf.argmax(logits, axis=-1)
    predictions = tf.cast(predictions, tf.int32)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32) * mask) / tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32), accuracy

# 模型初始化时创建层
class MyModel(tf.keras.Model):
    def __init__(self, generator):
        super(MyModel, self).__init__()
        self.generator = generator
        self.style_embeddings_layer = tf.keras.layers.Embedding(input_dim=2, output_dim=generator.config.n_embd)
    def call(self, styles, *args, **kwargs):
        style_embeddings = self.style_embeddings_layer(styles)
        return style_embeddings
    
# 数据集读取
def create_dataset(input_ids, attention_mask, labels, styles, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask, labels, styles))
    return dataset.shuffle(buffer_size=10000).batch(batch_size, drop_remainder=True)
