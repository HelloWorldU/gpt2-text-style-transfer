from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import pre_progress as pr
import discriminator as dis

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

def setup_optimizers():
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

    final_lr_schedule = WarmUpDecay(
        initial_learning_rate=initial_learning_rate,
        decay_schedule_fn=lr_schedule,
        warmup_steps=warmup_steps
    )

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=final_lr_schedule)
    dis_Y_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    dis_Z_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    return gen_optimizer, dis_Y_optimizer, dis_Z_optimizer, final_lr_schedule

accumulated_gradients = None

# 自定义训练步骤类
"""
in this task, we only trained X->Y.
"""
class Trainstep:
    def __init__(self, gen, gen_optimizer, dis_Y, dis_Y_optimizer, dis_Z, dis_Z_optimizer, tokenizer, embedding, strategy, final_lr_schedule):
        self.strategy = strategy
        self.gen = gen
        self.gen_optimizer = gen_optimizer
        self.dis_Y = dis_Y
        self.dis_Y_optimizer = dis_Y_optimizer
        self.dis_Z = dis_Z
        self.dis_Z_optimizer = dis_Z_optimizer
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.final_lr_schedule = final_lr_schedule


    # @tf.function
    def train_generator_step(self, input_ids, attention_mask, labels, styles, max_len, step, accumulation_steps=4, 
                             lambda_rec=1.0, lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
        global accumulated_gradients
        def step_fn(input_ids=input_ids, attention_mask=attention_mask, labels=labels, styles=styles, max_len=max_len, 
        accumulation_steps=accumulation_steps, lambda_rec=lambda_rec, lambda_lm=lambda_lm, lambda_adv=lambda_adv, lambda_kl=lambda_kl, gamma=gamma):
            with tf.GradientTape() as tape:
                tf.debugging.enable_check_numerics()
                
                accumulation_steps, lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma = pr.conv_tensor_to_float(accumulation_steps, lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma)

                epsilon = 1e-6 # 快速修复

                # 嵌入风格标签
                style_embeddings = self.embedding(styles) # [num_devices, batch_size, n_embd]
                print("Style embeddings shape:", style_embeddings.shape)  # Debug info
                
                # 将输入 ID 嵌入到相同的嵌入空间
                input_embeddings = self.gen.transformer.wte(input_ids) # [num_devices, batch_size, seq_len, n_embd]
                print("Input embeddings shape:", input_embeddings.shape)  # Debug info
                
                extended_input_embeddings = input_embeddings + tf.expand_dims(style_embeddings, axis=1)
                print("Extended embeddings shape:", extended_input_embeddings.shape)  # Debug info

                """
                at here, we need to remove the leading dimension
                """
                extended_input_embeddings = tf.squeeze(extended_input_embeddings, axis=0)
                input_ids, attention_mask, labels, styles = dis.convert_tensor(input_ids, attention_mask, labels, styles)
                input_ids, attention_mask = pr.remove_leading_dim(input_ids, attention_mask)

                outputs = self.gen(input_ids=input_ids, attention_mask=attention_mask, training=True)
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

                for var in self.gen.trainable_variables:
                    tf.debugging.check_numerics(var, message="Model weight check")

                print("Input shapes:", 
                        "input_ids:", input_ids.shape, 
                        "attention_mask:", attention_mask.shape, 
                        "labels:", labels.shape, 
                        "styles:", styles.shape,
                        "max_len:", max_len)

                actual_shape = tf.shape(input_ids)
                print("Actual shape:", actual_shape)  # Debug info
                
                max_new_tokens = tf.maximum(max_len - actual_shape[1] - 1, 0)
                batch_size = actual_shape[0]

                # 扩展 input_ids
                """
                at here, we need to padding to -> [batch_size, max_new_tokens]
                """
                padding = tf.zeros((batch_size, max_new_tokens), dtype=input_ids.dtype)
                print("Padding shape:", padding.shape)  # Debug info
                
                extended_input_ids = tf.concat([input_ids, padding], axis=1)
                extended_attention_mask1 = tf.concat([attention_mask, tf.zeros((attention_mask.shape[0],
                                            max_new_tokens), dtype=attention_mask.dtype)], axis=1)
                
                print(f"Extended input_ids shape: {extended_input_ids.shape}")
                print(f"Extended attention_mask shape: {extended_attention_mask1.shape}")

                try:
                    generated_ids = self.gen.generate(
                        extended_input_ids, 
                        attention_mask=extended_attention_mask1, 
                        max_length=max_len,  # 或者其他固定值
                        min_length=input_ids.shape[1],
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        bos_token_id=self.tokenizer.bos_token_id,
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
                
                generated_embeddings = self.gen.transformer.wte(generated_ids) # [batch_size, seq_len, n_embd]
                extended_generated_embeddings = generated_embeddings + tf.expand_dims(style_embeddings, axis=1)
                print("Generated extended embeddings shape:", extended_generated_embeddings.shape)  # Debug info

                padded_input_ids = tf.pad(input_ids, [[0, 0], [0, generated_ids.shape[1] - input_ids.shape[1]]],
                                        "CONSTANT", constant_values=self.tokenizer.pad_token_id)
                zx_distribution = dis.compute_distribution(extended_input_embeddings, self.dis_Z)
                zy_distribution = dis.compute_distribution(extended_generated_embeddings, self.dis_Z)
                kl_loss = dis.kl_divergence(zx_distribution, zy_distribution)
                print("KL loss:", kl_loss)  # Debug info

                extended_attention_mask2 = tf.pad(attention_mask, [[0, 0],[0, generated_ids.shape[1] - attention_mask.shape[1]]], 
                                            "CONSTANT", constant_values=1)
                real_lm_loss_Y = dis.compute_lm_loss(generated_ids, extended_attention_mask2, self.dis_Y)
                lm_loss = gamma * real_lm_loss_Y
                print("LM loss:", lm_loss)  # Debug info

                real_adv_loss = self.dis_Y(padded_input_ids, attention_mask=extended_attention_mask2, training=True)
                generated_adv_loss = self.dis_Y(generated_ids, attention_mask=extended_attention_mask2, training=True)

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
                gradients = tape.gradient(total_loss, self.gen.trainable_variables)
                gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=0.9)

                # 数值稳定性检查
                for grad in gradients:
                    tf.debugging.check_numerics(grad, "Gradient check")
            
            print("Gradients calculated")  # Debug info
            return total_loss, rec_loss, lm_loss, adv_loss, kl_loss, gradients, accuracy

        
        if accumulated_gradients is None:
            accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in self.gen.trainable_variables]
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
        self.gen_optimizer.apply_gradients(zip(accumulated_gradients, self.gen.trainable_variables))
        print("Gradients applied")  # Debug info
        current_lr = self.final_lr_schedule(step)
        
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
    # @tf.function
    def reinforce_step(self, input_ids, attention_mask, labels, style_ids, max_len):
        with tf.GradientTape() as tape:
            print("input_ids shape:", input_ids.shape)
            print("attention_mask shape:", attention_mask.shape)
            print("style_ids shape:", style_ids.shape)
            style_ids = tf.expand_dims(style_ids, axis=1)
            print("Max_length:", max_len)
            print("Fixed max length:", tf.get_static_value(max_len))

            """
            we also need to remove the leading dimension
            """
            input_ids, attention_mask = pr.remove_leading_dim(input_ids, attention_mask)

            generated_ids = self.gen.generate(input_ids, attention_mask=attention_mask, max_length=max_len)
            print("Generated IDs shape:", generated_ids.shape)
            generated_attention_mask = tf.pad(attention_mask, [[0, 0], [0, generated_ids.shape[1] - attention_mask.shape[1]]], "CONSTANT", constant_values=1)
            lm_loss = dis.compute_lm_loss(generated_ids, generated_attention_mask, self.dis_Y)
            
            outputs = self.gen(input_ids, attention_mask=attention_mask, labels=labels, training=False)
            logits = outputs.logits
            log_probs = tf.nn.log_softmax(logits, axis=-1)
            seq_len = tf.shape(generated_ids)[1]
            mask = tf.sequence_mask(tf.reduce_sum(tf.cast(generated_ids != self.gen.config.pad_token_id, tf.int32), axis=1), maxlen=seq_len)
            log_probs_padded = tf.pad(log_probs, [[0, 0], [0, tf.shape(mask)[1] - tf.shape(log_probs)[1]], [0, 0]], "CONSTANT", constant_values=0)
            masked_log_probs = tf.cast(log_probs_padded, tf.float32) * tf.cast(mask[:, :, tf.newaxis], tf.float32)
            log_prob = tf.reduce_mean(masked_log_probs) / tf.reduce_sum(tf.cast(mask, tf.float32))
            
            # 添加基线
            baseline = tf.reduce_mean(lm_loss)
            advantage = lm_loss - baseline
            loss = -tf.cast(advantage, tf.float32) * log_prob
        gradients = tape.gradient(loss, self.gen.trainable_variables)
        print("reinforce gradients calculated")  # Debug info
        self.gen_optimizer.apply_gradients(zip(gradients, self.gen.trainable_variables))
        return loss


    # 自定义鉴别器训练步骤
    # @tf.function
    def train_discriminator_Y_step(self, real_ids, real_mask, input_ids, attention_mask, max_len, gamma=1.0):
        """
        remove the leading dimension
        """
        real_ids, real_mask = pr.remove_leading_dim(real_ids, real_mask)
        input_ids, attention_mask = pr.remove_leading_dim(input_ids, attention_mask)

        generated_ids = self.gen.generate(input_ids, attention_mask=attention_mask, max_length=max_len)
        print(f"generated_ids.shape {generated_ids.shape} and real_ids.shape {real_ids.shape} and real_mask.shape {real_mask.shape}")
        with tf.GradientTape() as tape:
            real_loss = dis.compute_lm_loss(real_ids, real_mask, self.dis_Y)
            generated_mask = tf.pad(real_mask, [[0, 0], [0, generated_ids.shape[1] - real_ids.shape[1]]], "CONSTANT", constant_values=1)
            fake_loss = dis.compute_lm_loss(generated_ids, generated_mask, self.dis_Y)
            total_loss = real_loss - gamma * fake_loss
        gradients = tape.gradient(total_loss, self.dis_Y.trainable_variables)
        print("dis_Y gradients calculated")  # Debug info
        self.dis_Y_optimizer.apply_gradients(zip(gradients, self.dis_Y.trainable_variables))
        return total_loss, real_loss, fake_loss


    # 鉴别器z对齐
    # @tf.function    
    def discriminator_Z_loss(self, zx, zy, label_smoothing=0.1):
        print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=label_smoothing, reduction='none')
        zx = self.gen.transformer.wte(zx) # [batch_size, seq_len, n_embd]
        zy = self.dis_Y.transformer.wte(zy) # [batch_size, seq_len, n_embd]
        print(f"zx shape: {zx.shape}, zy shape: {zy.shape}")  # 调试信息
        zx_loss = bce(tf.ones_like(self.dis_Z(zx)), self.dis_Z(zx))
        zy_loss = bce(tf.zeros_like(self.dis_Z(zy)), self.dis_Z(zy))
        print("Discriminator Z loss calculated")  # 调试信息
        return zx_loss + zy_loss

    # 训练鉴别器z
    # @tf.function
    def train_discriminator_Z_step(self, zx, zy, label_smoothing=0.1):
        zx = tf.pad(zx, [[0, 0], [0, zy.shape[1] - zx.shape[1]]], "CONSTANT", constant_values=0)
        with tf.GradientTape() as tape:
            loss = self.discriminator_Z_loss(zx, zy, label_smoothing=label_smoothing)
        gradients = tape.gradient(loss, self.dis_Z.trainable_variables)
        print("Discriminator Z gradients calculated")  # Debug info
        self.dis_Z_optimizer.apply_gradients(zip(gradients, self.dis_Z.trainable_variables))
        return loss


    # 生成样本训练鉴别器z
    # @tf.function
    def train_generator_with_discriminator_Z(self, input_ids_X, input_ids_Y, attention_mask_X, attention_mask_Y, 
                                            style_ids_X, style_ids_Y, max_len, label_smoothing=0.1):
        style_ids_X = tf.expand_dims(style_ids_X, axis=1)
        style_ids_Y = tf.expand_dims(style_ids_Y, axis=1)

        """
        remove the leading dimension
        """
        input_ids_X, attention_mask_X = pr.remove_leading_dim(input_ids_X, attention_mask_X)
        input_ids_Y, attention_mask_Y = pr.remove_leading_dim(input_ids_Y, attention_mask_Y)

        generated_ids_X_Z = self.gen.generate(input_ids_X, attention_mask=attention_mask_X, max_length=max_len)
        generated_ids_Y_Z = self.gen.generate(input_ids_Y, attention_mask=attention_mask_Y, max_length=max_len)
        disc_z_loss_X = self.train_discriminator_Z_step(input_ids_X, generated_ids_X_Z, label_smoothing=label_smoothing)
        disc_z_loss_Y = self.train_discriminator_Z_step(input_ids_Y, generated_ids_Y_Z, label_smoothing=label_smoothing)
        return disc_z_loss_X, disc_z_loss_Y

