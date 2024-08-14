from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import numpy as np
import experiment as ex
import discriminator as dis

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)  # 禁用 XLA

# 日志调试
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 紧急执行
# tf.config.run_functions_eagerly(True)

# 混合精度
from tensorflow.keras.mixed_precision import set_global_policy
# set_global_policy('mixed_float16')

# 加载预训练模型和分词器
model_path = "./repository/"
tokenizer = BertTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
generator = TFGPT2LMHeadModel.from_pretrained(model_path)
discriminator_Y = TFGPT2LMHeadModel.from_pretrained(model_path)
discriminator_Z = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

special_tokens_dict = {'pad_token': '[PAD]'}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')  # 通常是0
tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('[SEP]')  # 通常是102
tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('[CLS]')  # 通常是101

generator.config.pad_token_id = tokenizer.pad_token_id

# 调整模型的词嵌入大小
generator.resize_token_embeddings(len(tokenizer))
discriminator_Y.resize_token_embeddings(len(tokenizer))

# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_X = os.path.join(current_dir, './corpus/X_corpus.txt')
file_path_Y = os.path.join(current_dir, './corpus/Y_corpus.txt')
test_file_path = os.path.join(current_dir, './corpus/test_corpus.txt')

debug_file_path = os.path.join(current_dir, './experiment/debug.txt')

# 加载数据集
train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, test_dataset = ex.load_dataset(file_path_X, file_path_Y, 
                                                                                                    test_file_path, tokenizer, seed=42)

# 使用自定义学习率调度器
initial_learning_rate = 1e-5
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

mymodel = ex.MyModel(generator)

# 自定义生成器训练步骤
# Only X->Y
@tf.function
def train_generator_step(generator, discriminator_Y, input_ids, attention_mask, labels, styles, max_len, step, accumulation_steps=4, lambda_rec=1.0, 
                         lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
    global accumulated_gradients
    def step_fn(input_ids=input_ids, attention_mask=attention_mask, labels=labels, styles=styles, max_len=max_len):
        with tf.GradientTape() as tape:
            tf.debugging.enable_check_numerics()
            
            epsilon = 1e-6 # 快速修复

            # 嵌入风格标签
            style_embeddings = mymodel(styles)
            print("Style embeddings shape:", style_embeddings.shape)  # Debug info
            
            # 将输入 ID 嵌入到相同的嵌入空间
            input_embeddings = generator.transformer.wte(input_ids) # [batch_size, seq_len, n_embd]
            print("Input embeddings shape:", input_embeddings.shape)  # Debug info
            
            extended_input_embeddings = input_embeddings + tf.expand_dims(style_embeddings, axis=1)
            print("Extended embeddings shape:", extended_input_embeddings.shape)  # Debug info

            input_ids, attention_mask, labels, styles = dis.convert_to_tensor(input_ids, attention_mask, labels, styles)

            outputs = generator(input_ids=input_ids, attention_mask=attention_mask, training=True)
            logits = outputs.logits
            print("Logits shape:", logits.shape)  # Debug info
            print("labels shape:", labels.shape)  # Debug info

            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
            mask = tf.cast(labels != -100, logits.dtype)
            print("Mask shape:", mask.shape)  # Debug info
    
            # Check for NaN or Inf in logits
            tf.debugging.check_numerics(logits, "Logits contain NaN or Inf")

            # 各个损失
            rec_loss = loss_fn(tf.where(labels == -100, tf.zeros_like(labels, dtype=logits.dtype), tf.cast(labels, logits.dtype)), logits)
            rec_loss = tf.reduce_sum(rec_loss * mask) / (tf.reduce_sum(mask) + epsilon)
            print("Reconstruction loss:", rec_loss)  # Debug info

            for var in generator.trainable_variables:
                tf.debugging.check_numerics(var, message="Model weight check")

            print("Input shapes:", 
                     "input_ids:", input_ids.shape, 
                     "attention_mask:", attention_mask.shape, 
                     "labels:", labels.shape, 
                     "styles:", styles.shape,
                     "max_len:", max_len)

            max_new_tokens = max_len - input_ids.shape[1] - 10
            batch_size = input_ids.shape[0]

            # 扩展 input_ids
            padding = tf.zeros((batch_size, max_new_tokens), dtype=input_ids.dtype)
            extended_input_ids = tf.concat([input_ids, padding], axis=1)
            extended_attention_mask1 = tf.concat([attention_mask, tf.zeros((attention_mask.shape[0],
                     max_new_tokens), dtype=attention_mask.dtype)], axis=1)
            
            print(f"Extended input_ids shape: {extended_input_ids.shape}")
            print(f"Extended attention_mask shape: {extended_attention_mask1.shape}")

            try:
                generated_ids = generator.generate(
                    extended_input_ids, 
                    attention_mask=extended_attention_mask1, 
                    max_length=max_len,  # 或者其他固定值
                    min_length=input_ids.shape[1],
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    bos_token_id=tokenizer.bos_token_id,
                    # use_cache=True,
                    # num_beams=1,  # 使用贪婪搜索
                    do_sample=False,  # 不使用采样
                    # temperature=1.0,  # 降低随机性
                )
                print("Generation successful. Generated IDs shape:", generated_ids.shape)

            except Exception as e:
                print(f"Error during generation: {e}")
                print(f"input_ids shape: {input_ids.shape}")
                print(f"attention_mask shape: {attention_mask.shape}")
                print(f"max_len: {max_len}")
                raise
            
            generated_embeddings = generator.transformer.wte(generated_ids) # [batch_size, seq_len, n_embd]
            extended_generated_embeddings = generated_embeddings + tf.expand_dims(style_embeddings, axis=1)
            print("Generated extended embeddings shape:", extended_generated_embeddings.shape)  # Debug info

            padded_input_ids = tf.pad(input_ids, [[0, 0], [0, generated_ids.shape[1] - input_ids.shape[1]]],
                                       "CONSTANT", constant_values=tokenizer.pad_token_id)
            zx_distribution = dis.compute_distribution(extended_input_embeddings, discriminator_Z)
            zy_distribution = dis.compute_distribution(extended_generated_embeddings, discriminator_Z)
            kl_loss = dis.kl_divergence(zx_distribution, zy_distribution)
            print("KL loss:", kl_loss)  # Debug info

            extended_attention_mask2 = tf.pad(attention_mask, [[0, 0],[0, generated_ids.shape[1] - attention_mask.shape[1]]], 
                                        "CONSTANT", constant_values=1)
            real_lm_loss_Y = dis.compute_lm_loss(generated_ids, extended_attention_mask2, discriminator_Y)
            lm_loss = gamma * real_lm_loss_Y
            print("LM loss:", lm_loss)  # Debug info

            real_adv_loss = discriminator_Y(padded_input_ids, attention_mask=extended_attention_mask2, training=True)
            generated_adv_loss = discriminator_Y(generated_ids, attention_mask=extended_attention_mask2, training=True)

            real_adv_loss_logits = tf.clip_by_value(real_adv_loss.logits, -1e6, 1e6)
            generated_adv_loss_logits = tf.clip_by_value(generated_adv_loss.logits, -1e6, 1e6)

            adv_loss = dis.compute_adversarial_loss(real_adv_loss_logits, generated_adv_loss_logits)
            print("Adversarial loss:", adv_loss)  # Debug info

            total_loss = lambda_rec * rec_loss - lambda_lm * lm_loss + lambda_adv * adv_loss + lambda_kl * kl_loss
            print("Total loss:", total_loss)  # Debug info

            predictions = tf.argmax(logits, axis=-1)
            predictions = tf.cast(predictions, tf.int32)
            mask = tf.cast(mask, tf.float32)
            accuracy = tf.reduce_sum(tf.cast(tf.equal(predictions, labels), tf.float32) * mask) / tf.reduce_sum(mask)
            print("Accuracy:", accuracy)  # Debug info

            # 梯度裁剪
            gradients = tape.gradient(total_loss, generator.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.9)

            # 数值稳定性检查
            for grad in gradients:
                tf.debugging.check_numerics(grad, "Gradient check")
        
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
    print("Gradients applied")  # Debug info
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
    print(f"{generated_ids.shape} and {real_ids.shape} and {real_mask.shape}")
    with tf.GradientTape() as tape:
        real_loss = dis.compute_lm_loss(real_ids, real_mask, model)
        generated_mask = tf.pad(real_mask, [[0, 0], [0, generated_ids.shape[1] - real_ids.shape[1]]], "CONSTANT", constant_values=1)
        fake_loss = dis.compute_lm_loss(generated_ids, generated_mask, model)
        total_loss = real_loss - gamma * fake_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    print("Discriminator_y gradients calculated")  # Debug info
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, real_loss, fake_loss

# 鉴别器z对齐
def discriminator_z_loss(discriminator_Z, zx, zy, label_smoothing=0.1):
    print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)
    zx = generator.transformer.wte(zx) # [batch_size, seq_len, n_embd]
    zy = discriminator_Y.transformer.wte(zy) # [batch_size, seq_len, n_embd]
    print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
    zx_loss = bce(tf.ones_like(discriminator_Z(zx)), discriminator_Z(zx))
    zy_loss = bce(tf.zeros_like(discriminator_Z(zy)), discriminator_Z(zy))
    print("Discriminator Z loss calculated")  # 调试信息
    return zx_loss + zy_loss

@tf.function
def train_discriminator_z_step(discriminator_Z, zx, zy, optimizer, label_smoothing):
    zx = tf.pad(zx, [[0, 0], [0, zy.shape[1] - zx.shape[1]]], "CONSTANT", constant_values=0)
    with tf.GradientTape() as tape:
        loss = discriminator_z_loss(discriminator_Z, zx, zy, label_smoothing=label_smoothing)
    gradients = tape.gradient(loss, discriminator_Z.trainable_variables)
    print("Discriminator Z gradients calculated")  # Debug info
    optimizer.apply_gradients(zip(gradients, discriminator_Z.trainable_variables))
    return loss

# 生成样本训练鉴别器z
def train_generator_with_discriminator_z(generator, discriminator_Z, input_ids_X, input_ids_Y, attention_mask_X, attention_mask_Y, 
                                         style_ids_X, style_ids_Y, max_len, generator_optimizer, discriminator_z_optimizer, label_smoothing=0.1):
    style_ids_X = tf.expand_dims(style_ids_X, axis=1)
    style_ids_Y = tf.expand_dims(style_ids_Y, axis=1)
    generated_ids_X_Z = generator.generate(input_ids_X, attention_mask=attention_mask_X, max_length=max_len)
    generated_ids_Y_Z = generator.generate(input_ids_Y, attention_mask=attention_mask_Y, max_length=max_len)
    disc_z_loss_X = train_discriminator_z_step(discriminator_Z, input_ids_X, generated_ids_X_Z, discriminator_z_optimizer, label_smoothing=label_smoothing)
    disc_z_loss_Y = train_discriminator_z_step(discriminator_Z, input_ids_Y, generated_ids_Y_Z, discriminator_z_optimizer, label_smoothing=label_smoothing)
    return disc_z_loss_X, disc_z_loss_Y

# REINFORCE算法
@tf.function
def reinforce_step(input_ids, attention_mask, labels, style_ids, max_len, generator, language_model, generator_optimizer):
    with tf.GradientTape() as tape:
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("style_ids shape:", style_ids.shape)
        style_ids = tf.expand_dims(style_ids, axis=1)
        print("Fixed max_length:", max_len)
        generated_ids = generator.generate(input_ids, attention_mask=attention_mask, max_length=max_len)
        generated_attention_mask = tf.pad(attention_mask, [[0, 0], [0, generated_ids.shape[1] - attention_mask.shape[1]]], "CONSTANT", constant_values=1)
        lm_loss = dis.compute_lm_loss(generated_ids, generated_attention_mask, language_model)
        
        outputs = generator(input_ids, attention_mask=attention_mask, labels=labels, training=False)
        logits = outputs.logits
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        seq_len = tf.shape(generated_ids)[1]
        mask = tf.sequence_mask(tf.reduce_sum(tf.cast(generated_ids != generator.config.pad_token_id, tf.int32), axis=1), maxlen=seq_len)
        log_probs_padded = tf.pad(log_probs, [[0, 0], [0, tf.shape(mask)[1] - tf.shape(log_probs)[1]], [0, 0]], "CONSTANT", constant_values=0)
        masked_log_probs = tf.cast(log_probs_padded, tf.float32) * tf.cast(mask[:, :, tf.newaxis], tf.float32)
        log_prob = tf.reduce_mean(masked_log_probs) / tf.reduce_sum(tf.cast(mask, tf.float32))
        
        # 添加基线
        baseline = tf.reduce_mean(lm_loss)
        advantage = lm_loss - baseline
        loss = -tf.cast(advantage, tf.float32) * log_prob
    gradients = tape.gradient(loss, generator.trainable_variables)
    print("reinforce gradients calculated")  # Debug info
    generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
    return loss

# 训练参数
epochs = 1
batch_size = 2
max_len = 0
accumulation_steps = 2
lambda_rec = 1.0
lambda_lm = 1.0
lambda_adv = 1.0
lambda_kl = 1.0
gamma = 1.0
label_smoothing = 0.1
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

# 创建输入数据
train_tf_dataset_X = ex.create_tf_dataset(train_dataset_X, batch_size)
train_tf_dataset_Y = ex.create_tf_dataset(train_dataset_Y, batch_size)
valid_tf_dataset_X = ex.create_tf_dataset(valid_dataset_X, batch_size)
valid_tf_dataset_Y = ex.create_tf_dataset(valid_dataset_Y, batch_size)
test_tf_dataset = ex.create_tf_dataset(test_dataset, batch_size)

# 加载鉴别器模型参数
discriminator_z_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
disc_steps = 2

# 训练循环
lr_step = 0
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} started")

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

    # 生成器鉴别器交替训练
    for batch_X, batch_Y in zip(train_tf_dataset_X, train_tf_dataset_Y):
        # 对于 X 数据
        batch_input_ids_X = batch_X['input_ids']
        batch_attention_mask_X = batch_X['attention_mask']
        batch_labels_X = ex.create_labels(batch_input_ids_X, batch_attention_mask_X)
        batch_styles_X = batch_X['style']

        # 对于 Y 数据
        batch_input_ids_Y = batch_Y['input_ids']
        batch_attention_mask_Y = batch_Y['attention_mask']
        batch_labels_Y = ex.create_labels(batch_input_ids_Y, batch_attention_mask_Y)
        batch_styles_Y = batch_Y['style']
        print("Processing batch")
        # print("batch_labels_X:", batch_labels_X)

        # 动态 Padding
        max_len_X, batch_input_ids_X, batch_attention_mask_X = dis.dynamic_padding(batch_input_ids_X, batch_attention_mask_X)
        max_len_Y, batch_input_ids_Y, batch_attention_mask_Y = dis.dynamic_padding(batch_input_ids_Y, batch_attention_mask_Y)
        max_len = max(max_len_X, max_len_Y) + 101
        
        batch_attention_mask_X = tf.convert_to_tensor(batch_attention_mask_X, dtype=tf.int32)
        batch_attention_mask_Y = tf.convert_to_tensor(batch_attention_mask_Y, dtype=tf.int32)

        print("batch_input_ids_X shape:", batch_input_ids_X.shape)
        print("batch_attention_mask_X shape:", batch_attention_mask_X.shape)
        print("batch_input_ids_Y shape:", batch_input_ids_Y.shape)
        print("batch_attention_mask_Y.shape:", batch_attention_mask_Y.shape)

        # 生成器
        print("Training generator")
        gen_loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy = train_generator_step(generator, discriminator_Y, 
                                                                                batch_input_ids_X, batch_attention_mask_X, batch_labels_X, 
                                                                                batch_styles_X, max_len, tf.cast(lr_step, tf.float32), 
                                                                                accumulation_steps, lambda_rec, lambda_lm, lambda_adv, 
                                                                                lambda_kl, gamma)
        print("Training REINFORCE")
        reinforce_loss = reinforce_step(batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, max_len, generator, 
                                                                            discriminator_Y, generator_optimizer)

        # 鉴别器
        generated_ids_Y = generator.generate(batch_input_ids_X, attention_mask=batch_attention_mask_X, max_length=max_len)
        print("Training discriminator Y")
        disc_loss_Y, real_loss_Y, fake_loss_Y = train_discriminator_y_step(batch_input_ids_Y, batch_attention_mask_Y, generated_ids_Y, 
                                                                           discriminator_Y, discriminator_y_optimizer)
        print("Training discriminator Z")
        disc_z_loss_X, disc_z_loss_Y = train_generator_with_discriminator_z(generator, discriminator_Z, batch_input_ids_X, batch_input_ids_Y, 
                                                                            batch_attention_mask_X, batch_attention_mask_Y, 
                                                                            batch_styles_X, batch_styles_Y, max_len, generator_optimizer, 
                                                                            discriminator_z_optimizer, label_smoothing)
        print("Epoch completed")
        
    total_gen_loss += tf.cast(gen_loss, tf.float32)
    print("Generator loss: ", gen_loss)
    total_rec_loss += tf.cast(rec_loss, tf.float32)
    total_lm_loss += tf.cast(lm_loss, tf.float32)
    total_adv_loss += tf.cast(adv_loss, tf.float32)
    total_kl_loss += tf.cast(kl_loss, tf.float32)
    total_disc_loss += tf.cast(disc_loss_Y, tf.float32)
    total_disc_z_loss += tf.cast((disc_z_loss_X + disc_z_loss_Y) / 2, tf.float32)
    total_accuracy += tf.cast(accuracy, tf.float32)
    total_reinforce_loss += tf.cast(reinforce_loss, tf.float32)
    num_batches += 1
    lr_step += 1
    learning_rates.append(current_lr.numpy())

    # 验证循环
    total_valid_loss = tf.constant(0.0, dtype=tf.float32)
    total_valid_accuracy = tf.constant(0.0, dtype=tf.float32)
    num_valid_batches = 0
    for batch_valid_X in valid_tf_dataset_X:
        batch_valid_ids = batch_valid_X['input_ids']
        batch_valid_attention_mask = batch_valid_X['attention_mask']
        batch_valid_labels = ex.create_labels(batch_valid_ids, batch_valid_attention_mask)
        batch_valid_styles = batch_valid_X['style']

        batch_valid_attention_mask = tf.convert_to_tensor(batch_valid_attention_mask, dtype=tf.int32)
        loss, accuracy = ex.valid_step(generator, batch_valid_ids, batch_valid_attention_mask, batch_valid_labels, batch_valid_styles)
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
    avg_reinforce_loss = total_reinforce_loss / tf.cast(num_batches, tf.float32)
    avg_valid_loss = total_valid_loss / tf.cast(num_valid_batches, tf.float32)
    avg_valid_accuracy = total_valid_accuracy / tf.cast(num_valid_batches, tf.float32)
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
    print(f"Epoch {epoch + 1} ended")
    print(f"Train Loss: {avg_train_loss:.4f}, Valid Loss: {avg_valid_loss:.4f}")
    print(f"Train Accuracy: {avg_train_accuracy:.4f}, Valid Accuracy: {avg_valid_accuracy:.4f}")
    print(f"Train Perplexity: {train_perplexity:.4f}, Valid Perplexity: {valid_perplexity:.4f}")
    print(f"Discriminator Loss: {avg_disc_loss:.4f}")
    print(f"REINFORCE Loss: {avg_reinforce_loss:.4f}")
    print("---")

# 测试集评估
test_accuracies, test_losses, test_perplexity = ex.test_evalution(generator, ex.test_step,test_tf_dataset)

# 保存绘图
save_dir = './experiment'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ex.perplexity_curve(perplexities, save_dir)
ex.loss_curve(train_losses, valid_losses, test_losses, save_dir)
ex.accuracy_curve(train_accuracies, valid_accuracies, test_accuracies, save_dir)
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
