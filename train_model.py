from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import experiment as ex
import discriminator as dis
import pre_progress as pr
import logging

# 日志级别
logging.basicConfig(level=logging.INFO)

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
train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, \
test_dataset = ex.load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, seed=42)

accumulated_gradients = None

mymodel = ex.MyModel(generator)

# 自定义生成器训练步骤
# Only X->Y
@tf.function
def train_generator_step(gen, gen_optimizer, dis_Y, dis_Z, input_ids, attention_mask, labels, styles, max_len, step, final_lr_schedule,
                         accumulation_steps=4, lambda_rec=1.0, lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
    global accumulated_gradients
    def step_fn(input_ids=input_ids, attention_mask=attention_mask, labels=labels, styles=styles, max_len=max_len):
        with tf.GradientTape() as tape:
            tf.debugging.enable_check_numerics()
            
            epsilon = 1e-6 # 快速修复

            # 嵌入风格标签
            style_embeddings = mymodel(styles)
            print("Style embeddings shape:", style_embeddings.shape)  # Debug info
            
            # 将输入 ID 嵌入到相同的嵌入空间
            input_embeddings = gen.transformer.wte(input_ids) # [batch_size, seq_len, n_embd]
            print("Input embeddings shape:", input_embeddings.shape)  # Debug info
            
            extended_input_embeddings = input_embeddings + tf.expand_dims(style_embeddings, axis=1)
            print("Extended embeddings shape:", extended_input_embeddings.shape)  # Debug info

            input_ids, attention_mask, labels, styles = dis.convert_to_tensor(input_ids, attention_mask, labels, styles)

            outputs = gen(input_ids=input_ids, attention_mask=attention_mask, training=True)
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

            for var in gen.trainable_variables:
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
                generated_ids = gen.generate(
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
            
            generated_embeddings = gen.transformer.wte(generated_ids) # [batch_size, seq_len, n_embd]
            extended_generated_embeddings = generated_embeddings + tf.expand_dims(style_embeddings, axis=1)
            print("Generated extended embeddings shape:", extended_generated_embeddings.shape)  # Debug info

            padded_input_ids = tf.pad(input_ids, [[0, 0], [0, generated_ids.shape[1] - input_ids.shape[1]]],
                                       "CONSTANT", constant_values=tokenizer.pad_token_id)
            zx_distribution = dis.compute_distribution(extended_input_embeddings, dis_Z)
            zy_distribution = dis.compute_distribution(extended_generated_embeddings, dis_Z)
            kl_loss = dis.kl_divergence(zx_distribution, zy_distribution)
            print("KL loss:", kl_loss)  # Debug info

            extended_attention_mask2 = tf.pad(attention_mask, [[0, 0],[0, generated_ids.shape[1] - attention_mask.shape[1]]], 
                                        "CONSTANT", constant_values=1)
            real_lm_loss_Y = dis.compute_lm_loss(generated_ids, extended_attention_mask2, dis_Y)
            lm_loss = gamma * real_lm_loss_Y
            print("LM loss:", lm_loss)  # Debug info

            real_adv_loss = dis_Y(padded_input_ids, attention_mask=extended_attention_mask2, training=True)
            generated_adv_loss = dis_Y(generated_ids, attention_mask=extended_attention_mask2, training=True)

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
            gradients = tape.gradient(total_loss, gen.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.9)

            # 数值稳定性检查
            for grad in gradients:
                tf.debugging.check_numerics(grad, "Gradient check")
        
        print("Gradients calculated")  # Debug info
        return total_loss, rec_loss, lm_loss, adv_loss, kl_loss, gradients, accuracy

    
    if accumulated_gradients is None:
        accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in gen.trainable_variables]
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
    gen_optimizer.apply_gradients(zip(accumulated_gradients, gen.trainable_variables))
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
def train_discriminator_Y_step(real_ids, real_mask, generated_ids, model, optimizer, gamma=1.0):
    print(f"{generated_ids.shape} and {real_ids.shape} and {real_mask.shape}")
    with tf.GradientTape() as tape:
        real_loss = dis.compute_lm_loss(real_ids, real_mask, model)
        generated_mask = tf.pad(real_mask, [[0, 0], [0, generated_ids.shape[1] - real_ids.shape[1]]], "CONSTANT", constant_values=1)
        fake_loss = dis.compute_lm_loss(generated_ids, generated_mask, model)
        total_loss = real_loss - gamma * fake_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    print("dis_Y gradients calculated")  # Debug info
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return total_loss, real_loss, fake_loss

# 鉴别器z对齐
def discriminator_Z_loss(gen, dis_Y, dis_Z, zx, zy, label_smoothing=0.1):
    print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing)
    zx = gen.transformer.wte(zx) # [batch_size, seq_len, n_embd]
    zy = dis_Y.transformer.wte(zy) # [batch_size, seq_len, n_embd]
    print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
    zx_loss = bce(tf.ones_like(dis_Z(zx)), dis_Z(zx))
    zy_loss = bce(tf.zeros_like(dis_Z(zy)), dis_Z(zy))
    print("Discriminator Z loss calculated")  # 调试信息
    return zx_loss + zy_loss

@tf.function
def train_discriminator_Z_step(dis_Z, zx, zy, optimizer, label_smoothing):
    zx = tf.pad(zx, [[0, 0], [0, zy.shape[1] - zx.shape[1]]], "CONSTANT", constant_values=0)
    with tf.GradientTape() as tape:
        loss = discriminator_Z_loss(dis_Z, zx, zy, label_smoothing=label_smoothing)
    gradients = tape.gradient(loss, dis_Z.trainable_variables)
    print("Discriminator Z gradients calculated")  # Debug info
    optimizer.apply_gradients(zip(gradients, dis_Z.trainable_variables))
    return loss

# 生成样本训练鉴别器z
def train_generator_with_discriminator_Z(gen, dis_Z, input_ids_X, input_ids_Y, attention_mask_X, attention_mask_Y, 
                                         style_ids_X, style_ids_Y, max_len, gen_optimizer, dis_z_optimizer, label_smoothing=0.1):
    style_ids_X = tf.expand_dims(style_ids_X, axis=1)
    style_ids_Y = tf.expand_dims(style_ids_Y, axis=1)
    generated_ids_X_Z = gen.generate(input_ids_X, attention_mask=attention_mask_X, max_length=max_len)
    generated_ids_Y_Z = gen.generate(input_ids_Y, attention_mask=attention_mask_Y, max_length=max_len)
    disc_z_loss_X = train_discriminator_Y_step(dis_Z, input_ids_X, generated_ids_X_Z, dis_z_optimizer, label_smoothing=label_smoothing)
    disc_z_loss_Y = train_discriminator_Y_step(dis_Z, input_ids_Y, generated_ids_Y_Z, dis_z_optimizer, label_smoothing=label_smoothing)
    return disc_z_loss_X, disc_z_loss_Y

# REINFORCE算法
@tf.function
def reinforce_step(input_ids, attention_mask, labels, style_ids, max_len, gen, language_model, gen_optimizer):
    with tf.GradientTape() as tape:
        print("input_ids shape:", input_ids.shape)
        print("attention_mask shape:", attention_mask.shape)
        print("style_ids shape:", style_ids.shape)
        style_ids = tf.expand_dims(style_ids, axis=1)
        print("Fixed max_length:", max_len)
        generated_ids = gen.generate(input_ids, attention_mask=attention_mask, max_length=max_len)
        generated_attention_mask = tf.pad(attention_mask, [[0, 0], [0, generated_ids.shape[1] - attention_mask.shape[1]]], "CONSTANT", constant_values=1)
        lm_loss = dis.compute_lm_loss(generated_ids, generated_attention_mask, language_model)
        
        outputs = gen(input_ids, attention_mask=attention_mask, labels=labels, training=False)
        logits = outputs.logits
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        seq_len = tf.shape(generated_ids)[1]
        mask = tf.sequence_mask(tf.reduce_sum(tf.cast(generated_ids != gen.config.pad_token_id, tf.int32), axis=1), maxlen=seq_len)
        log_probs_padded = tf.pad(log_probs, [[0, 0], [0, tf.shape(mask)[1] - tf.shape(log_probs)[1]], [0, 0]], "CONSTANT", constant_values=0)
        masked_log_probs = tf.cast(log_probs_padded, tf.float32) * tf.cast(mask[:, :, tf.newaxis], tf.float32)
        log_prob = tf.reduce_mean(masked_log_probs) / tf.reduce_sum(tf.cast(mask, tf.float32))
        
        # 添加基线
        baseline = tf.reduce_mean(lm_loss)
        advantage = lm_loss - baseline
        loss = -tf.cast(advantage, tf.float32) * log_prob
    gradients = tape.gradient(loss, gen.trainable_variables)
    print("reinforce gradients calculated")  # Debug info
    gen_optimizer.apply_gradients(zip(gradients, gen.trainable_variables))
    return loss

# 训练参数
trainconfig = ex.Trainconfig()

# 创建输入数据
train_tf_dataset_X = ex.create_tf_dataset(train_dataset_X, trainconfig.batch_size)
train_tf_dataset_Y = ex.create_tf_dataset(train_dataset_Y, trainconfig.batch_size)
valid_tf_dataset_X = ex.create_tf_dataset(valid_dataset_X, trainconfig.batch_size)
valid_tf_dataset_Y = ex.create_tf_dataset(valid_dataset_Y, trainconfig.batch_size)
test_tf_dataset = ex.create_tf_dataset(test_dataset, trainconfig.batch_size)


# 训练
class Train(tf.keras.Model):
    def __init__(self, gen, dis_Y, dis_Z, config):
        super(Train, self).__init__()
        self.gen = gen
        self.dis_Y = dis_Y
        self.dis_Z = dis_Z
        self.cfg = config
        self.setup_optimizers()
        self.setup_metrics()

    def setup_optimizers(self):
        """
        we use a polynomial decay learning rate schedule with warmup here.
        """
        initial_learning_rate = 1e-5
        warmup_steps = 8 # 5%
        decay_steps = 152
        end_learning_rate = 1e-6

        lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            end_learning_rate=end_learning_rate,
            power=1.0
        )

        self.final_lr_schedule = ex.WarmUpDecay(
            initial_learning_rate=initial_learning_rate,
            decay_schedule_fn=lr_schedule,
            warmup_steps=warmup_steps
        )

        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate=self.final_lr_schedule)
        self.dis_Y_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
        self.dis_Z_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    def setup_metrics(self):
        metric_names = ['train_loss', 'train_accuracy', 'rec_loss', 'lm_loss', 'adv_loss', 
                        'kl_loss', 'disc_y_loss', 'disc_z_loss', 'reinforce_loss', 
                        'valid_loss', 'valid_accuracy', 'learning_rate', 'perplexity']
        self.metrics = {name: tf.keras.metrics.Mean(name=name) for name in metric_names}

    def update_metrics(self, **kwargs) -> dict:
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].update_state(value)

    def reset_metrics(self):
        for metric in self.metrics.value():
            metric.reset_states()

    def train(self, train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, epochs, *args, **kwargs):
        for epoch in range(epochs):
            self.reset_metrics()

            for batch_X, batch_Y in zip(train_tf_dataset_X, train_tf_dataset_Y):
                print(f"Train Epoch {epoch + 1} started")

                batch_input_ids_X = batch_X['input_ids']
                batch_attention_mask_X = batch_X['attention_mask']
                batch_labels_X = ex.create_labels(batch_input_ids_X, batch_attention_mask_X)
                batch_styles_X = batch_X['style']

                batch_input_ids_Y = batch_Y['input_ids']
                batch_attention_mask_Y = batch_Y['attention_mask']
                batch_labels_Y = ex.create_labels(batch_input_ids_Y, batch_attention_mask_Y)
                batch_styles_Y = batch_Y['style']
                print("Processing batch")

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

                print("Training gen")
                loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr \
                , accuracy = train_generator_step(self.gen, self.gen_optimizer, 
                            self.dis_Y, self.dis_Z, batch_input_ids_X, 
                            batch_attention_mask_X, batch_labels_X, 
                            batch_styles_X, max_len, tf.cast(self.cfg.lr_step, tf.float32), 
                            self.final_lr_schdule, self.cfg.accumulation_steps,
                            self.cfg.lambda_rec, self.cfg.lambda_lm, 
                            self.cfg.lambda_adv, self.cfg.lambda_kl, 
                            self.cfg.gamma)
                
                print("Training REINFORCE")
                reinforce_loss = reinforce_step(batch_input_ids_X, 
                                batch_attention_mask_X, batch_labels_X, 
                                batch_styles_X, max_len, self.gen, 
                                self.dis_Y, self.gen_optimizer)

                generated_ids_Y = self.gen.generate(batch_input_ids_X, attention_mask=batch_attention_mask_X, max_length=max_len)

                print("Training discriminator Y")
                disc_loss_Y, real_loss_Y \
                , fake_loss_Y = train_discriminator_Y_step(batch_input_ids_Y, 
                                batch_attention_mask_Y, generated_ids_Y, 
                                self.dis_Y, self.dis_Y_optimizer, self.cfg.gamma)
                
                print("Training discriminator Z")
                disc_z_loss_X, disc_z_loss_Y = train_generator_with_discriminator_Z(
                                self.gen, self.dis_Z, batch_input_ids_X, 
                                batch_input_ids_Y, batch_attention_mask_X, 
                                batch_attention_mask_Y, batch_styles_X, 
                                batch_styles_Y, max_len, self.gen_optimizer, 
                                self.dis_z_optimizer, self.cfg.label_smoothing)
                
                self.cfg.lr_step += 1
                self.num_batches += 1
                print(f"Train Epoch {epoch + 1} completed")

            print(f"Valid Epoch {epoch + 1} started")
            for batch_valid_X in valid_tf_dataset_X:
                batch_valid_ids = batch_valid_X['input_ids']
                batch_valid_attention_mask = batch_valid_X['attention_mask']
                batch_valid_labels = ex.create_labels(batch_valid_ids, batch_valid_attention_mask)
                batch_valid_styles = batch_valid_X['style']

                batch_valid_attention_mask = tf.convert_to_tensor(batch_valid_attention_mask, dtype=tf.int32)
                valid_loss, valid_accuracy = ex.valid_step(self.gen, 
                                    batch_valid_ids, batch_valid_attention_mask, 
                                    batch_valid_labels, batch_valid_styles)
                self.total_valid_loss += loss
                self.total_valid_accuracy += accuracy
                num_valid_batches += 1

                self.update_metrics(loss, accuracy, rec_loss, lm_loss, adv_loss, kl_loss,
                            disc_loss_Y, (disc_z_loss_X + disc_z_loss_Y) / 2.0,
                            reinforce_loss, valid_loss, valid_accuracy, current_lr)
            
            self.print_epoch_results()
            print(f"Valid Epoch {epoch + 1} completed")

        self.metrics['valid_perplexity'] = tf.exp(self.metrics['valid_loss'].result())
        self.append_to_config()

    def append_to_config(self):
        self.cfg.train_losses.append(self.metrics['train_loss'].result().numpy())
        self.cfg.train_accuracies.append(self.metrics['train_accuracy'].result().numpy())
        self.cfg.rec_losses.append(self.metrics['rec_loss'].result().numpy())
        self.cfg.lm_losses.append(self.metrics['lm_loss'].result().numpy())
        self.cfg.adv_losses.append(self.metrics['adv_loss'].result().numpy())
        self.cfg.kl_losses.append(self.metrics['kl_loss'].result().numpy())
        self.cfg.disc_losses.append(self.metrics['disc_y_loss'].result().numpy())
        self.cfg.disc_z_losses.append(self.metrics['disc_z_loss'].result().numpy())
        self.cfg.reinforce_losses.append(self.metrics['reinforce_loss'].result().numpy())
        self.cfg.valid_losses.append(self.metrics['valid_loss'].result().numpy())
        self.cfg.valid_accuracies.append(self.metrics['valid_accuracy'].result().numpy())
        self.cfg.perplexities.append(self.metrics['valid_perplexity'].result().numpy())
        self.cfg.learning_rates.append(self.metrics['current_lr'].result().numpy())

    def print_epoch_results(self):
        template_key = {name for name in self.metrics.keys()}
        template_value = {metric.result().numpy() for metric in self.metrics.values()}
        template = ''.join([f"{key}: {value} "for key, value in zip(template_key, template_value)])
        for _, result in enumerate(template):
            print(result)

# 实例化并训练
train_model = Train(generator, discriminator_Y, discriminator_Z, trainconfig)

import json

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        "chief": ["host1:2222"],
        "worker": ["host2:2222", "host3:2222", "host4:2222"]
    },
    "task": {"type": "worker", "index": 1}
})

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    train_model.train(train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, trainconfig.epochs)

# 测试集评估
test_accuracies, test_losses, test_perplexity = ex.test_evalution(generator, ex.test_step,test_tf_dataset)

# 保存绘图
save_dir = './experiment'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ex.perplexity_curve(train_model.cfg.perplexities, save_dir)
ex.loss_curve(train_model.cfg.train_losses, train_model.cfg.valid_losses, save_dir)
ex.accuracy_curve(train_model.cfg.train_accuracies, train_model.cfg.valid_accuracies, save_dir)
ex.learning_rate_curve(train_model.cfg.learning_rates, save_dir)
ex.plot_losses(train_model.cfg.rec_losses, train_model.cfg.lm_losses, train_model.cfg.adv_losses,
                train_model.cfg.kl_losses, train_model.cfg.disc_losses, train_model.cfg.disc_z_losses, save_dir)

# 保存模型和分词器
generator.save_pretrained('./model/generator')
discriminator_Y.save_pretrained('./model/discriminator_Y')
discriminator_Z.save_weights('./model/discriminator_Z/discriminator_Z_weights')
tokenizer.save_pretrained('./model/tokenizer')

# 生成文本
prompts = ["我应该是听说过的。", "我想，我眼见你慢慢倒地，怎么会摔坏呢，装腔作势罢了，真是可恶。"]
with open('./corpus/generated.txt', 'w', encoding='utf-8') as f:
    for prompt in prompts:
        generated_text = ex.generate_text(generator, tokenizer, prompt)
        f.write(generated_text + "\n")
