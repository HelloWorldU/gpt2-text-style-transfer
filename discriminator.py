from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

# 损失函数
def compute_perplexity(logits, labels, tokenizer):
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != tokenizer.pad_token_id, tf.float32)
    loss = loss_fn(labels, logits)
    loss = tf.reduce_sum(loss * mask, axis=-1) / tf.reduce_sum(mask, axis=-1)
    perplexity = tf.exp(loss)
    return perplexity

def compute_discriminator_loss(real_texts, generated_texts, model, tokenizer, gamma=0):
    real_inputs = tokenizer(real_texts, return_tensors='tf', padding=True, truncation=True)
    generated_inputs = tokenizer(generated_texts, return_tensors='tf', padding=True, truncation=True)
    real_outputs = model(real_inputs['input_ids'], attention_mask=real_inputs['attention_mask'], training=False)
    generated_outputs = model(generated_inputs['input_ids'], attention_mask=generated_inputs['attention_mask'], training=False)
    real_perplexity = compute_perplexity(real_outputs.logits, real_inputs['input_ids'], tokenizer)
    generated_perplexity = compute_perplexity(generated_outputs.logits, generated_inputs['input_ids'], tokenizer)
    loss_real = -tf.reduce_mean(tf.math.log(real_perplexity))
    loss_generated = -tf.reduce_mean(tf.math.log(generated_perplexity))
    return loss_real + gamma * loss_generated

def compute_lm_loss(input_ids, real_mask, language_model):
    outputs = language_model(input_ids, attention_mask=real_mask, training=True)
    logits = outputs.logits
    shift_logits = logits[:,:-1,:]
    shift_labels = input_ids[:,1:]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(shift_labels, shift_logits)
    loss = tf.clip_by_value(loss, 1e-8, 1e8)
    return tf.reduce_mean(loss)

# 鉴别器数据处理函数
def encode_texts(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='tf', max_length=max_length, padding='max_length', truncation=True)
    return inputs['input_ids'], inputs['attention_mask']

# KL散度正则化
def kl_divergence(zx_distribution, zy_distribution):
    epsilon = 1e-8
    kl_loss = tf.reduce_mean(zx_distribution * tf.math.log((zx_distribution + epsilon) / (zy_distribution + epsilon)))
    return tf.clip_by_value(kl_loss, -1e3, 1e3)

# 定义计算内容分布的函数
def compute_distribution(input_ids, discriminator_Z):
    outputs = discriminator_Z(input_ids)
    content_vector = tf.reduce_mean(outputs, axis=1)
    return content_vector

# 动态padding
def pad_sequences(sequences, padding_val=0):
    max_len = max(len(seq) for seq in sequences)
    return [seq + [padding_val] * (max_len - len(seq)) for seq in sequences]

def create_attention_mask(sequences):
    return [[1 if token !=0 else 0 for token in seq] for seq in sequences]

def dynamic_padding(batch_input_ids, batch_attention_mask):
    max_len = max(len(seq) for seq in batch_input_ids)
    print("max_len:", max_len)
    padded_input_ids = tf.keras.preprocessing.sequence.pad_sequences(batch_input_ids, maxlen=max_len, padding='post')
    # print("attention_mask:", batch_attention_mask)
    padded_attention_mask = tf.keras.preprocessing.sequence.pad_sequences(batch_attention_mask, maxlen=max_len, padding='post')
    # print("later attention_mask:", padded_attention_mask)
    return max_len, padded_input_ids, padded_attention_mask

# 对抗损失
def compute_adversarial_loss(real_logits, generated_logits):
    # 使用sigmoid交叉熵损失
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    
    # 真实样本的目标是1，生成样本的目标是0
    real_loss = loss_fn(tf.ones_like(real_logits), real_logits)
    generated_loss = loss_fn(tf.zeros_like(generated_logits), generated_logits)
    
    # 计算平均损失
    total_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    
    return total_loss

# 处理输入张量
def convert_to_tensor(input_ids, attention_mask, labels, styles):
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    styles = tf.convert_to_tensor(styles, dtype=tf.int32)
    return input_ids, attention_mask, labels, styles

# 显示生成id
def generate_with_float32(generator, input_ids, attention_mask, **kwargs):
    input_ids = tf.cast(input_ids, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    # Use the new mixed precision policy
    original_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    try:
        generated_ids = generator.generate(input_ids, attention_mask=attention_mask, **kwargs)
        generated_ids = tf.cast(generated_ids, tf.float32)
    finally:
        # Restore the original policy
        tf.keras.mixed_precision.set_global_policy(original_policy)
    return generated_ids
