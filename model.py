from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf

# 加载预训练模型和分词器
def create_model():
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

    return generator, discriminator_Y, discriminator_Z, tokenizer


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


accumulated_gradients = None

# 自定义生成器训练步骤
"""
in this task, we only trained X->Y.
"""
@tf.function
def train_generator_step(gen, gen_optimizer, dis_Y, dis_Z, input_ids, attention_mask, labels, styles, max_len, step, final_lr_schedule,
                         mymodel, accumulation_steps=4, lambda_rec=1.0, lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
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

# 训练鉴别器z
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

