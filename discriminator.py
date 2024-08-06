from transformers import GPT2Tokenizer, TFGPT2LMHeadModel
import tensorflow as tf

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

def compute_lm_loss(input_ids,language_model):
    outputs=language_model(input_ids,training=False)
    logits=outputs.logits
    shift_logits=logits[:,:-1,:]
    shift_labels=input_ids[:,1:]
    loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')
    loss=loss_fn(shift_labels,shift_logits)
    return tf.recude_mean(loss)

# 鉴别器数据处理函数
def encode_texts(texts, tokenizer, max_length=512):
    inputs = tokenizer(texts, return_tensors='tf', max_length=max_length, padding='max_length', truncation=True)
    return inputs['input_ids'], inputs['attention_mask']