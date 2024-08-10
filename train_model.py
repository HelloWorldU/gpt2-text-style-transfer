from transformers import BertTokenizer, TFGPT2LMHeadModel, DataCollatorForLanguageModeling
import tensorflow as tf
import os
import experiment as ex
import discriminator as dis

# 加载预训练模型和分词器
model_path = "./repository/"
tokenizer = BertTokenizer.from_pretrained(model_path)
generator = TFGPT2LMHeadModel.from_pretrained(model_path)
discriminator_Y = TFGPT2LMHeadModel.from_pretrained(model_path)
discriminator_Z = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

special_tokens_dict = {'pad_token': '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

# 调整模型的词嵌入大小
generator.resize_token_embeddings(len(tokenizer))
discriminator_Y.resize_token_embeddings(len(tokenizer))

# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_X = os.path.join(current_dir, './corpus/X_corpus.txt')
file_path_Y = os.path.join(current_dir, './corpus/Y_corpus.txt')
test_file_path = os.path.join(current_dir, './corpus/test_corpus.txt')

# 加载数据集
(train_dataset_X, train_styles_X), (train_dataset_Y, train_styles_Y), (valid_dataset_X, valid_styles_X), (
    valid_dataset_Y, valid_styles_Y), (test_dataset, test_styles) = ex.load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, seed=42)

# 创建输入数据
train_input_ids_X, train_attention_mask_X, train_labels_X, train_styles_X = ex.prepare_input_data(train_dataset_X, train_styles_X)
train_input_ids_Y, train_attention_mask_Y, train_labels_Y, train_styles_Y = ex.prepare_input_data(train_dataset_Y, train_styles_Y)
valid_input_ids_X, valid_attention_mask_X, valid_labels_X, valid_styles_X = ex.prepare_input_data(valid_dataset_X, valid_styles_X)
valid_input_ids_Y, valid_attention_mask_Y, valid_labels_Y, valid_styles_Y = ex.prepare_input_data(valid_dataset_Y, valid_styles_Y)

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

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=final_lr_schedule)

accumulated_gradients = None

# 自定义生成器训练步骤
# Only X->Y
# @tf.function
def train_generator_step(generator, discriminator_Y, input_ids, attention_mask, labels, styles, step, accumulation_steps=4, lambda_rec=1.0, 
                         lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
    global accumulated_gradients
    def step_fn():
        with tf.GradientTape() as tape:
            # 嵌入风格标签
            style_embeddings = tf.keras.layers.Embedding(input_dim=2, output_dim=generator.config.n_embd)(styles)
            style_embeddings = tf.expand_dims(style_embeddings, axis=1) # [batch_size, 1, n_embd]
            print("Style embeddings shape:", style_embeddings.shape)  # Debug info
            
            # 将输入 ID 嵌入到相同的嵌入空间
            input_embeddings = generator.transformer.wte(input_ids) # [batch_size, seq_len, n_embd]
            print("Input embeddings shape:", input_embeddings.shape)  # Debug info
            
            extended_embeddings = tf.concat([style_embeddings, input_embeddings], axis=1)
            print("Extended embeddings shape:", extended_embeddings.shape)  # Debug info

            style_attention_mask = tf.ones((styles.shape[0], 1), dtype=attention_mask.dtype) # [batch_size, 1]
            print("Style attention mask shape:", style_attention_mask.shape)  # Debug info
            extended_attention_mask = tf.concat([style_attention_mask, attention_mask], axis=1)
            print("Extended attention mask shape:", extended_attention_mask.shape)  # Debug info

            outputs = generator(extended_embeddings, attention_mask=extended_attention_mask, training=True)
            logits = outputs.logits
            print("Logits shape:", logits.shape)  # Debug info

            extended_labels = tf.pad(labels, [[0,0], [1,0]], constant_values=-100)
            print("Extended labels shape:", extended_labels.shape)  # Debug info

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mask = tf.cast(extended_labels != -100, logits.dtype)
            print("Mask shape:", mask.shape)  # Debug info

            # 各个损失
            rec_loss = loss_fn(tf.where(extended_labels == -100, 0, extended_labels), logits)
            rec_loss = tf.reduce_sum(rec_loss * mask) / tf.reduce_sum(mask)
            print("Reconstruction loss:", rec_loss)  # Debug info

            generated_ids = generator.generate(extended_embeddings, max_length=input_ids.shape[1]+20)
            print("Generated IDs shape:", generated_ids.shape)  # Debug info

            zx_distribution = ex.compute_distribution(extended_embeddings, discriminator_Z)
            zy_distribution = ex.compute_distribution(generated_ids, discriminator_Z)
            kl_loss = ex.kl_divergence(zx_distribution, zy_distribution)
            print("KL loss:", kl_loss)  # Debug info

            real_lm_loss_Y = dis.compute_lm_loss(generated_ids, extended_attention_mask, discriminator_Y)
            lm_loss = gamma * real_lm_loss_Y
            print("LM loss:", lm_loss)  # Debug info

            real_adv_loss = discriminator_Y(extended_embeddings, attention_mask=extended_attention_mask, training=True)
            generated_adv_loss = discriminator_Y(generated_ids, attention_mask=attention_mask, training=True)
            adv_loss = real_adv_loss + generated_adv_loss
            print("Adversarial loss:", adv_loss)  # Debug info

            total_loss = lambda_rec * rec_loss - lambda_lm * lm_loss + lambda_adv * adv_loss + lambda_kl * kl_loss
            print("Total loss:", total_loss)  # Debug info

            predictions = tf.argmax(logits, axis=-1)
            predictions = tf.cast(predictions, tf.int32)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32) * mask) / tf.reduce_sum(mask)
            print("Accuracy:", accuracy)  # Debug info
            
        gradients = tape.gradient(total_loss, generator.trainable_variables)
        print("Gradients calculated")  # Debug info
        return total_loss, rec_loss, lm_loss, adv_loss, kl_loss, gradients, accuracy

    
    if accumulated_gradients is None:
        accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in generator.trainable_variables]
    total_loss = tf.constant(0.0, dtype=tf.float32)
    total_rec_loss = tf.constant(0.0, dtype=tf.float32)
    total_lm_loss = tf.constant(0.0, dtype=tf.float32)
    total_adv_loss = tf.constant(0.0, dtype=tf.float32)
    total_kl_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    
    for _ in range(accumulation_steps):
        step_total_loss, step_rec_loss, step_lm_loss, step_adv_loss, step_kl_loss, step_gradients, step_accuracy = step_fn()
        total_loss += tf.cast(step_total_loss,tf.float32)
        total_rec_loss += tf.cast(step_rec_loss, tf.float32)
        total_lm_loss += tf.cast(step_lm_loss, tf.float32)
        total_adv_loss += tf.cast(step_adv_loss, tf.float32)
        total_kl_loss += tf.cast(step_kl_loss, tf.float32)
        total_accuracy += step_accuracy
        for i, g in enumerate(step_gradients):
            accumulated_gradients[i].assign_add(g)
    
    for i, g in enumerate(accumulated_gradients):
        accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))
    generator_optimizer.apply_gradients(zip(accumulated_gradients, generator.trainable_variables))
    current_lr = final_lr_schedule(step)
    
    for grad in accumulated_gradients:
        grad.assign(tf.zeros_like(grad))

    return (total_loss / tf.cast(accumulation_steps, tf.float32),
            total_rec_loss / tf.cast(accumulation_steps, tf.float32),
            total_lm_loss / tf.cast(accumulation_steps, tf.float32),
            total_adv_loss / tf.cast(accumulation_steps, tf.float32),
            total_kl_loss / tf.cast(accumulation_steps, tf.float32),
            current_lr,
            total_accuracy / tf.cast(accumulation_steps, tf.float32))


# 自定义鉴别器训练步骤
@tf.function
def train_discriminator_y_step(real_ids, real_mask, generated_ids, model, optimizer, gamma=1.0):
    with tf.GradientTape() as tape:
        real_loss = dis.compute_lm_loss(real_ids, real_mask, model)
        fake_loss = dis.compute_lm_loss(generated_ids, real_mask, model)
        total_loss = real_loss - gamma * fake_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, real_loss, fake_loss

# 鉴别器z对齐
def discriminator_z_loss(discriminator_Z, zx, zy, label_smoothing=0.1):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=label_smoothing)
    zx_loss = bce(tf.ones_like(discriminator_Z(zx)), discriminator_Z(zx))
    zy_loss = bce(tf.zeros_like(discriminator_Z(zy)), discriminator_Z(zy))
    return zx_loss + zy_loss

@tf.function
def train_discriminator_z_step(discriminator_Z, zx, zy, optimizer, label_smoothing):
    with tf.GradientTape() as tape:
        loss = discriminator_z_loss(discriminator_Z, zx, zy, label_smoothing=label_smoothing)
    gradients = tape.gradient(loss, discriminator_Z.trainable_variables)
    optimizer.apply_gradients(zip(gradients, discriminator_Z.trainable_variables))
    return loss

# 生成样本训练鉴别器z
def train_generator_with_discriminator_z(generator, discriminator_Z, input_ids_X, input_ids_Y, attention_mask_X, attention_mask_Y, 
                                         style_ids_X, style_ids_Y, generator_optimizer, discriminator_z_optimizer, label_smoothing=0.1):
    generated_ids_X_Z = generator.generate(input_ids_X, attention_mask=attention_mask_X, max_length=input_ids_X.shape[1] + style_ids_X.shape[1] + 20)
    generated_ids_Y_Z = generator.generate(input_ids_Y, attention_mask=attention_mask_Y, max_length=input_ids_Y.shape[1] + style_ids_Y.shape[1] + 20)
    disc_z_loss_X = train_discriminator_z_step(discriminator_Z, input_ids_X, generated_ids_X_Z, discriminator_z_optimizer, label_smoothing)
    disc_z_loss_Y = train_discriminator_z_step(discriminator_Z, input_ids_Y, generated_ids_Y_Z, discriminator_z_optimizer, label_smoothing)
    return disc_z_loss_X, disc_z_loss_Y

# REINFORCE算法
@tf.function
def reinforce_step(input_ids, attention_mask, labels, style_ids, generator, language_model, generator_optimizer):
    with tf.GradientTape() as tape:
        generated_ids = generator.generate(input_ids, attention_mask=attention_mask, max_length=input_ids.shape[1] + style_ids.shape[1] + 20)
        lm_loss = dis.compute_lm_loss(generated_ids, language_model)
        # 计算生成器生成文本的对数概率
        outputs = generator(input_ids, attention_mask=attention_mask, labels=labels, training=False)
        logits = outputs.logits
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        mask = tf.sequence_mask(labels, maxlen=tf.shape(logits)[-2])
        masked_log_probs = tf.boolean_mask(log_probs, mask)
        log_prob = tf.reduce_mean(masked_log_probs)
        loss = -lm_loss  * log_prob
    gradients = tape.gradient(loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

# 训练参数
epochs = 10
batch_size = 2
accumulation_steps = 2
lambda_rec = 1.0
lambda_lm = 1.0
lambda_adv = 1.0
lambda_kl = 1.0
gamma = 1.0
train_losses = []
rec_losses = []
lm_losses = []
adv_losses = []
kl_losses = []
disc_losses = []
disc_z_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
perplexities = []
learning_rates = []

# 数据集
train_dataset_X = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_input_ids_X, tf.int32),
    tf.cast(train_attention_mask_X, tf.int32),
    tf.cast(train_labels_X, tf.int32),
    tf.cast(train_styles_X, tf.int32))
).batch(batch_size)
train_dataset_Y = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_input_ids_Y, tf.int32),
    tf.cast(train_attention_mask_Y, tf.int32),
    tf.cast(train_labels_Y, tf.int32),
    tf.cast(train_styles_Y, tf.int32))
).batch(batch_size)


valid_dataset_X = tf.data.Dataset.from_tensor_slices((
    tf.cast(valid_input_ids_X, tf.int32),
    tf.cast(valid_attention_mask_X, tf.int32),
    tf.cast(valid_labels_X, tf.int32),
    tf.cast(valid_styles_X, tf.int32))
).batch(batch_size)
valid_dataset_Y = tf.data.Dataset.from_tensor_slices((
    tf.cast(valid_input_ids_Y, tf.int32),
    tf.cast(valid_attention_mask_Y, tf.int32),
    tf.cast(valid_labels_Y, tf.int32),
    tf.cast(valid_styles_Y, tf.int32))
).batch(batch_size)


discriminator_dataset_Y = tf.data.Dataset.from_tensor_slices((
    tf.cast(train_input_ids_Y, tf.int32),
    tf.cast(train_attention_mask_Y, tf.int32),
    tf.cast(train_labels_Y, tf.int32),
    tf.cast(train_styles_Y, tf.int32))
).batch(batch_size)

# 加载鉴别器模型参数
discriminator_z_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
disc_steps = 2

# 训练循环
step = 0
for epoch in range(epochs):
    total_gen_loss = tf.constant(0.0, dtype=tf.float32)
    total_rec_loss = tf.constant(0.0, dtype=tf.float32)
    total_lm_loss = tf.constant(0.0, dtype=tf.float32)
    total_adv_loss = tf.constant(0.0, dtype=tf.float32)
    total_kl_loss = tf.constant(0.0, dtype=tf.float32)
    total_disc_loss = tf.constant(0.0, dtype=tf.float32)
    total_disc_z_loss = tf.constant(0.0, dtype=tf.float32)
    total_accuracy = tf.constant(0.0, dtype=tf.float32)
    total_reinforce_loss = tf.constant(0.0, dtype=tf.float32)
    num_batches = 0
    combined_dataset = zip(train_dataset_X, train_dataset_Y, discriminator_dataset_Y)
    # 生成器鉴别器交替训练
    for (batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X), (batch_input_ids_Y, batch_attention_mask_Y, batch_labels_Y, 
            batch_styles_Y), (batch_real_ids_Y,batch_real_mask_Y, batch_real_labels_Y, batch_real_styles_Y) in combined_dataset:
        
        # 生成器
        gen_loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy = train_generator_step(generator, discriminator_Y, 
            batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, tf.cast(step, tf.float32), 
            accumulation_steps, lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma)
        reinforce_loss = reinforce_step(batch_input_ids, batch_attention_mask, batch_labels_X,batch_styles_X, generator, 
            discriminator_Y, generator_optimizer)

        # 鉴别器
        generated_ids_Y = generator.generate(batch_real_ids_Y, attention_mask=batch_real_mask_Y, max_length=batch_real_ids_Y.shape[1]+20)
        disc_loss_Y, real_loss_Y, fake_loss_Y = train_discriminator_y_step(batch_real_ids_Y, batch_real_mask_Y, generated_ids_Y, discriminator_Y, discriminator_optimizer)
        
        disc_z_loss_X, disc_z_loss_Y = train_generator_with_discriminator_z(generator, discriminator_Z, batch_input_ids_X, batch_input_ids_Y, 
        batch_attention_mask_X, batch_attention_mask_Y, batch_styles_X, batch_styles_Y, generator_optimizer, discriminator_z_optimizer)
        
        total_gen_loss += gen_loss
        total_rec_loss += rec_loss
        total_lm_loss += lm_loss
        total_adv_loss += adv_loss
        total_kl_loss += kl_loss
        total_disc_loss += disc_loss_Y
        total_disc_z_loss += (disc_z_loss_X + disc_z_loss_Y) / 2
        total_accuracy += accuracy
        total_reinforce_loss += reinforce_loss
        num_batches += 1
        step += 1
        learning_rates.append(current_lr.numpy())

    # 验证循环
    total_valid_loss = tf.constant(0.0, dtype=tf.float32)
    total_valid_accuracy = tf.constant(0.0, dtype=tf.float32)
    num_valid_batches = 0
    for batch_input_ids, batch_attention_mask, batch_labels, batch_styles in valid_dataset_X:
        loss, accuracy = ex.valid_step(generator, batch_input_ids, batch_attention_mask, batch_labels, batch_styles)
        total_valid_loss += loss
        total_valid_accuracy += accuracy
        num_valid_batches += 1
    avg_train_loss = total_gen_loss / tf.cast(num_batches, tf.float32)
    avg_rec_loss = total_rec_loss / tf.cast(num_batches, tf.float32)
    avg_lm_loss = total_lm_loss / tf.cast(num_batches, tf.float32)
    avg_adv_loss = total_adv_loss / tf.cast(num_batches, tf.float32)
    avg_kl_loss = total_kl_loss / tf.cast(num_batches, tf.float32)
    avg_disc_loss = total_disc_loss / tf.cast(num_batches, tf.float32)
    avg_disc_z_loss = total_disc_z_loss / tf.cast(num_batches, tf.float32)
    avg_train_accuracy = total_accuracy / tf.cast(num_batches, tf.float32)
    avg_valid_loss = total_valid_loss / tf.cast(num_valid_batches, tf.float32)
    avg_valid_accuracy = total_valid_accuracy / tf.cast(num_valid_batches, tf.float32)
    avg_disc_loss = total_disc_loss / tf.cast(num_batches, tf.float32)
    avg_reinforce_loss = total_reinforce_loss / tf.cast(num_batches, tf.float32)
    train_losses.append(avg_train_loss.numpy())
    rec_losses.append(avg_rec_loss.numpy())
    lm_losses.append(avg_lm_loss.numpy())
    adv_losses.append(avg_adv_loss.numpy())
    kl_losses.append(avg_kl_loss.numpy())
    disc_losses.append(avg_disc_loss.numpy())
    disc_z_losses.append(avg_disc_z_loss.numpy())
    train_accuracies.append(avg_train_accuracy.numpy())
    valid_losses.append(avg_valid_loss.numpy())
    valid_accuracies.append(avg_valid_accuracy.numpy())
    train_perplexity = tf.exp(avg_train_loss).numpy()
    valid_perplexity = tf.exp(avg_valid_loss).numpy()
    perplexities.append(valid_perplexity)
    print(f"Epoch {epoch+1}")
    print(f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    print(f"Train Accuracy: {avg_train_accuracy:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
    print(f"Train Perplexity: {train_perplexity:.4f}, Valid Perplexity: {valid_perplexity:.4f}")
    print(f"Discriminator Loss: {avg_disc_loss:.4f}")
    print(f"REINFORCE Loss: {avg_reinforce_loss:.4f}")
    print("---")

# 测试集评估
text_input_ids, test_attention_mask, test_labels = ex.prepare_input_data(test_dataset)
test_dataset=tf.data.Dataset.from_tensor_slices((
    tf.cast(text_input_ids, tf.int32),
    tf.cast(test_attention_mask, tf.int32),
    tf.cast(test_labels, tf.int32),
    tf.cast(test_styles, tf.int32)
)).batch(batch_size)
ex.test_evalution(ex.test_step,test_dataset)

# 保存绘图
save_dir = './experiment'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ex.perplexity_curve(perplexities, save_dir)
ex.loss_curve(train_losses, valid_losses, save_dir)
ex.accuracy_curve(train_accuracies, valid_accuracies, save_dir)
ex.learning_rate_curve(learning_rates, save_dir)
ex.plot_losses(rec_losses, lm_losses, adv_losses, kl_losses, disc_losses, disc_z_losses, save_dir)

# 保存模型和分词器
generator.save_pretrained('./model/generator')
discriminator_Y.save_pretrained('./model/discriminator_Y')
discriminator_Z.save('./model/discriminator_Z')
tokenizer.save_pretrained('./model/tokenizer')

# 生成文本
prompts = ["我应该是听说过的。", "我想，我眼见你慢慢倒地，怎么会摔坏呢，装腔作势罢了，真是可恶。"]
with open('./corpus/generated.txt', 'w', encoding='utf-8') as f:
    for prompt in prompts:
        generated_text = ex.generate_text(generator, tokenizer, prompt)
        f.write(generated_text + "\n")
