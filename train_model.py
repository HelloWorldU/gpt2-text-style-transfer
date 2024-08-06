from transformers import BertTokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling
import tensorflow as tf
import os
import experiment as ex
import discriminator as dis

# 禁用 GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# # 混合精度训练
# from tensorflow.keras import mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_global_policy(policy)

# 加载预训练模型和分词器
model_name = "uer/gpt2-chinese-cluecorpussmall"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = TFGPT2LMHeadModel.from_pretrained(model_name)

# 添加 pad_token
special_tokens_dict = {'pad_token': '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 调整模型的词嵌入大小
model.resize_token_embeddings(len(tokenizer))

# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, './corpus/part_corpus.txt')
test_file_path = os.path.join(current_dir, './corpus/test_corpus.txt')

# 加载数据集
train_dataset, valid_dataset, test_dataset = ex.load_dataset(file_path, test_file_path, tokenizer, seed=42)

# 创建输入数据
train_input_ids, train_attention_mask, train_labels = ex.prepare_input_data(train_dataset)
valid_input_ids, valid_attention_mask, valid_labels = ex.prepare_input_data(valid_dataset)

# 加载鉴别器数据集
texts_X = ["源风格文本1", "源风格文本2", ...]
texts_Y = ["目标风格文本1", "目标风格文本2", ...]
input_ids_X, attention_mask_X = dis.encode_texts(texts_X, tokenizer)
input_ids_Y, attention_mask_Y = dis.encode_texts(texts_Y, tokenizer)

# 动态 Padding
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# 使用自定义学习率调度器
initial_learning_rate = 2e-4
warmup_steps = 8 # 5%
decay_steps = 152
end_learning_rate = 1e-6

# Decay schedule
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    end_learning_rate=end_learning_rate,
    power=1.0
)

final_lr_schedule = ex.WarmUpDecay(
    initial_learning_rate=initial_learning_rate,
    decay_schedule_fn=lr_schedule,
    warmup_steps=warmup_steps
)

optimizer = tf.keras.optimizers.Adam(learning_rate=final_lr_schedule)
# optimizer = mixed_precision.LossScaleOptimizer(optimizer)

accumulated_gradients = None

# 自定义训练验证和测试步骤
@tf.function
def train_step(input_ids, attention_mask, labels, step, accumulation_steps=4, lambda_rec=1.0, lambda_lm=1.0, gamma=1.0):
    global accumulated_gradients
    def step_fn():
        with tf.GradientTape() as tape:
            outputs = model(input_ids, attention_mask=attention_mask, training=True)
            logits = outputs.logits
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mask = tf.cast(labels != -100, logits.dtype)
            rec_loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
            rec_loss = tf.reduce_sum(rec_loss * mask) / tf.reduce_sum(mask)
            generated_ids = model.generate(input_ids, max_length=input_ids.shape[1]+20)
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            real_lm_loss = dis.compute_lm_loss(input_ids, model)
            generated_lm_loss = dis.compute_lm_loss(generated_ids, discriminator_model)
            lm_loss = real_lm_loss + gamma * generated_lm_loss
            total_loss = lambda_rec * rec_loss - lambda_lm * lm_loss
            predictions = tf.argmax(logits, axis=-1)
            predictions = tf.cast(predictions, tf.int32)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32) * mask) / tf.reduce_sum(mask)
        gradients = tape.gradient(total_loss, model.trainable_variables)
        return total_loss, rec_loss, lm_loss, gradients, accuracy
    if accumulated_gradients is None:
        accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in model.trainable_variables]
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_rec_loss = tf.constant(0.0, dtype=tf.float32)
    total_lm_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    for _ in range(accumulation_steps):
        step_total_loss, step_rec_loss, step_lm_loss, step_gradients, step_accuracy = step_fn()
        total_loss += tf.cast(step_total_loss,tf.float32)
        total_rec_loss += tf.cast(step_rec_loss, tf.float32)
        total_lm_loss += tf.cast(step_lm_loss, tf.float32)
        total_accuracy += step_accuracy
        for i, g in enumerate(step_gradients):
            accumulated_gradients[i].assign_add(g)
    for i, g in enumerate(accumulated_gradients):
        accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))
    optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
    current_lr = final_lr_schedule(step)
    for grad in accumulated_gradients:
        grad.assign(tf.zeros_like(grad))
    return (total_loss / tf.cast(accumulation_steps, tf.float32),total_rec_loss / tf.cast(accumulation_steps, tf.float32),total_lm_loss / tf.cast(accumulation_steps, tf.float32),current_lr,total_accuracy / tf.cast(accumulation_steps, tf.float32))

@tf.function
def valid_step(input_ids, attention_mask, labels):
    outputs = model(input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, logits.dtype)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    predictions=tf.argmax(logits, axis=-1)
    predictions=tf.cast(predictions,tf.int32)
    accuracy=tf.reduce_sum(tf.cast(tf.equal(predictions,labels),tf.float32)*mask)/tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32),accuracy

@tf.function
def test_step(input_ids, attention_mask, labels):
    outputs = model(input_ids, attention_mask=attention_mask, training=False)
    logits = outputs.logits
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    mask = tf.cast(labels != -100, logits.dtype)
    loss = loss_fn(tf.where(labels == -100, 0, labels), logits)
    loss = tf.reduce_sum(loss * mask) / tf.reduce_sum(mask)
    predictions = tf.argmax(logits, axis=-1)
    predictions = tf.cast(predictions, tf.int32)
    accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32)*mask)/tf.reduce_sum(mask)
    return tf.cast(loss, tf.float32), accuracy

# 自定义鉴别器训练步骤
@tf.function
def train_discriminator_step(generator_model, discriminator_model, input_ids, attention_mask, optimizer):
    with tf.GradientTape() as tape:
        generated_ids = generator_model.generate(input_ids, max_length=50)
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        real_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        real_perplexity = dis.compute_perplexity(real_texts, discriminator_model, tokenizer)
        generated_perplexity = dis.compute_perplexity(generated_texts, discriminator_model, tokenizer)
        loss = tf.reduce_mean(generated_perplexity) - tf.reduce_mean(real_perplexity)
    gradients = tape.gradient(loss, discriminator_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator_model.trainable_variables))
    return loss

# 训练循环
epochs = 10
batch_size = 16
accumulation_steps = 2
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
perplexities = []
learning_rates = []
train_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_input_ids, tf.int32),
    tf.cast(train_attention_mask, tf.int32),
    tf.cast(train_labels, tf.int32)
)).batch(batch_size)
valid_dataset = tf.data.Dataset.from_tensor_slices((
    tf.cast(valid_input_ids, tf.int32),
    tf.cast(valid_attention_mask, tf.int32),
    tf.cast(valid_labels, tf.int32)
)).batch(batch_size)

step = 0
for epoch in range(epochs):
    total_train_loss = tf.constant(0.0, dtype=tf.float32)
    total_train_accuracy = tf.constant(0.0, dtype=tf.float32)
    total_valid_loss = tf.constant(0.0, dtype=tf.float32)
    total_valid_accuracy = tf.constant(0.0, dtype=tf.float32)
    num_train_batches = 0
    num_valid_batches = 0
    for batch_input_ids, batch_attention_mask, batch_labels in train_dataset:
        loss, current_lr, accuracy = train_step(batch_input_ids, batch_attention_mask, batch_labels, tf.cast(step, tf.float32), accumulation_steps)
        total_train_loss += loss
        total_train_accuracy += accuracy
        num_train_batches += 1
        step += 1
        learning_rates.append(current_lr.numpy())
    for batch_input_ids, batch_attention_mask, batch_labels in valid_dataset:
        loss, accuracy = valid_step(batch_input_ids, batch_attention_mask, batch_labels)
        total_valid_loss +=loss
        total_valid_accuracy += accuracy
        num_valid_batches +=1
    avg_train_loss = total_train_loss / tf.cast(num_train_batches, tf.float32)
    avg_train_accuracy = total_train_accuracy / tf.cast(num_train_batches, tf.float32)
    avg_valid_loss = total_valid_loss / tf.cast(num_valid_batches, tf.float32)
    avg_valid_accuracy = total_valid_accuracy / tf.cast(num_valid_batches, tf.float32)
    train_losses.append(avg_train_loss.numpy())
    train_accuracies.append(avg_train_accuracy.numpy())
    valid_losses.append(avg_valid_loss.numpy())
    valid_accuracies.append(avg_valid_accuracy.numpy())
    train_perplexity=tf.exp(avg_train_loss).numpy()
    valid_perplexity=tf.exp(avg_valid_loss).numpy()
    perplexities.append(valid_perplexity)
    print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss}, Valid Loss: {avg_valid_loss}")
    print(f"Train Perplexity: {train_perplexity}, Valid Perplexity: {valid_perplexity}")

# 鉴别器数据集
discriminator_dataset = tf.data.Dataset.from_tensor_slices((input_ids_X, attention_mask_X)).shuffle(len(texts_X)).batch(batch_size)

# 加载鉴别器模型参数
discriminator_model = TFGPT2LMHeadModel.from_pretrained(model_name)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
disc_steps = 2

# 鉴别器训练
for epoch in range(epochs):
    for batch_input_ids, batch_attention_mask, batch_labels in discriminator_dataset:
        loss, current_lr, gen_loss, disc_loss = train_step(batch_input_ids, batch_attention_mask, batch_labels, discriminator_model, tf.cast(step, tf.float32), accumulation_steps)
        # ... 记录损失和其他指标 ...
        step += 1
    # 训练鉴别器
    for _ in range(disc_steps):  # 可以调整鉴别器的训练频率
        for batch_input_ids, batch_attention_mask, _ in discriminator_dataset:
            disc_loss = train_discriminator_step(model, batch_input_ids, batch_attention_mask, optimizer, discriminator_model)
        # ... 记录鉴别器损失 ...

# 测试集评估
text_input_ids, test_attention_mask, test_labels = ex.prepare_input_data(test_dataset)
test_dataset=tf.data.Dataset.from_tensor_slices((
    tf.cast(text_input_ids, tf.int32),
    tf.cast(test_attention_mask, tf.int32),
    tf.cast(test_labels, tf.int32)
)).batch(batch_size)
ex.test_evalution(test_step,test_dataset)

# 保存绘图
save_dir = './experiment'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ex.perplexity_curve(perplexities, save_dir)
ex.loss_curve(train_losses, valid_losses, save_dir)
ex.accuracy_curve(train_accuracies, valid_accuracies, save_dir)
ex.learning_rate_curve(learning_rates, save_dir)

# 保存模型和分词器
model.save_pretrained('./model/generrator')
discriminator_model.save_pretrained('./model/discriminator')
tokenizer.save_pretrained('./model')

# 生成文本
prompts = ["我应该是听说过的。", "我想，我眼见你慢慢倒地，怎么会摔坏呢，装腔作势罢了，真是可恶。"]
with open('./corpus/generated.txt', 'w', encoding='utf-8') as f:
    for prompt in prompts:
        generated_text = ex.generate_text(model, tokenizer, prompt)
        f.write(generated_text + "\n")
