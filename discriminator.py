from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf
# from tensorflow.keras.mixed_precision import set_global_policy

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
    input_ids = tf.cast(input_ids, tf.float32)
    real_mask = tf.cast(real_mask, tf.float32)
    outputs = language_model(input_ids, attention_mask=real_mask, training=True)
    logits = outputs.logits
    shift_logits = logits[:,:-1,:]
    shift_labels = input_ids[:,1:]
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    loss = loss_fn(shift_labels, shift_logits)
    loss = tf.cast(loss, tf.float32)
    loss = tf.clip_by_value(loss, 1e-8, 1e8)
    return tf.reduce_mean(loss)


# 鉴别器数据处理函数
def encode_texts(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='tf', max_length=max_length, padding='max_length', truncation=True)
    return inputs['input_ids'], inputs['attention_mask']


# 计算内容向量
def compute_distribution(input_embeddings, intermediate_model):
    input_embeddings = tf.cast(input_embeddings, tf.float32)
    
    outputs = intermediate_model(input_embeddings)
    
    content_vector = tf.reduce_mean(outputs, axis=1)
    
    print(f"Content vector shape: {content_vector.shape}")
    
    return content_vector


# KL散度正则化函数保持不变
def kl_divergence(zx_distribution, zy_distribution):
    epsilon = 1e-8
    zx_distribution = tf.cast(zx_distribution, tf.float32)
    zy_distribution = tf.cast(zy_distribution, tf.float32)

    safe_zx = tf.clip_by_value(zx_distribution, epsilon, 1.0 - epsilon)
    safe_zy = tf.clip_by_value(zy_distribution, epsilon, 1.0 - epsilon)

    try:
        safe_zx = tf.debugging.check_numerics(safe_zx, "NaN or Inf in safe_zx")
        safe_zy = tf.debugging.check_numerics(safe_zy, "NaN or Inf in safe_zy")

        kl_loss = tf.reduce_mean(safe_zx * tf.math.log(safe_zx / safe_zy))

        kl_loss = tf.debugging.check_numerics(kl_loss, "NaN or Inf in kl_loss")

        kl_loss =  tf.clip_by_value(kl_loss, -1e2, 1e2)
    except tf.errors.InvalidArgumentError as e:
        print("Exception caught:", e.message)
        print("safe_zx:", safe_zx.numpy())
        print("safe_zx:", safe_zy.numpy())
        raise
    return kl_loss


import tensorflow as tf

# 动态padding
@tf.function
def dynamic_padding(batch_input_ids, batch_attention_mask):
    strategy = tf.distribute.get_strategy()

    @tf.function
    def step_fn(batch_input_ids, batch_attention_mask):
        max_len = tf.reduce_max(tf.map_fn(lambda x: tf.shape(x)[0], batch_input_ids))
        padded_input_ids = tf.pad(batch_input_ids, [[0, 0], [0, max_len - tf.shape(batch_input_ids)[1]]])
        padded_attention_mask = tf.pad(batch_attention_mask, [[0, 0], [0, max_len - tf.shape(batch_attention_mask)[1]]])
        return max_len, padded_input_ids, padded_attention_mask

    per_replica_results = strategy.run(step_fn, args=(batch_input_ids, batch_attention_mask))
    max_len = tf.reduce_max(strategy.experimental_local_results(per_replica_results[0]))
    padded_input_ids = strategy.experimental_local_results(per_replica_results[1])
    padded_attention_mask = strategy.experimental_local_results(per_replica_results[2])
    
    return max_len, padded_input_ids, padded_attention_mask


# 对抗损失
def compute_adversarial_loss(real_logits, generated_logits):
    # 使用sigmoid交叉熵损失
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')
    
    # 真实样本的目标是1，生成样本的目标是0
    real_loss = tf.cast(loss_fn(tf.ones_like(real_logits), real_logits), tf.float32)
    generated_loss = tf.cast(loss_fn(tf.zeros_like(generated_logits), generated_logits), tf.float32)
    
    # 计算平均损失
    total_loss = tf.reduce_mean(real_loss) + tf.reduce_mean(generated_loss)
    
    return total_loss


# 处理输入张量
def convert_tensor(input_ids, attention_mask, labels, styles):
    input_ids = tf.convert_to_tensor(input_ids, dtype=tf.int32)
    attention_mask = tf.convert_to_tensor(attention_mask, dtype=tf.int32)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    styles = tf.convert_to_tensor(styles, dtype=tf.int32)
    return input_ids, attention_mask, labels, styles


# 显示生成id
def generate_with_float32(gen, input_ids, attention_mask, **kwargs):
    input_ids = tf.cast(input_ids, tf.int32)
    attention_mask = tf.cast(attention_mask, tf.int32)
    # Use the new mixed precision policy
    original_policy = tf.keras.mixed_precision.global_policy()
    tf.keras.mixed_precision.set_global_policy('float32')
    try:
        generated_ids = gen.generate(input_ids, attention_mask=attention_mask, **kwargs)
        generated_ids = tf.cast(generated_ids, tf.float32)
    finally:
        # Restore the original policy
        tf.keras.mixed_precision.set_global_policy(original_policy)
    return generated_ids
