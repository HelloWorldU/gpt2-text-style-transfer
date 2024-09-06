from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.mixed_precision import LossScaleOptimizer
import pre_progress as pr
import discriminator as dis

# 加载预训练模型和分词器
def create_model():
    model_path = "./repository/"
    tokenizer = BertTokenizer.from_pretrained(model_path, padding_side="left")
    generator = TFGPT2LMHeadModel.from_pretrained(model_path)
    discriminator_Y = TFGPT2LMHeadModel.from_pretrained(model_path)
    
    input_dim = 768

    inputs = Input(shape=(None, input_dim)) 
    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = GlobalAveragePooling1D()(x)  
    outputs = Dense(1)(x)

    discriminator_Z = Model(inputs=inputs, outputs=outputs)

    intermediate_outputs = discriminator_Z.get_layer('dense_2').output  
    intermediate_dis_Z = Model(inputs=inputs, outputs=intermediate_outputs)

    special_tokens_dict = {'pad_token': '[PAD]'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')  # 通常是0
    tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids('[SEP]')  # 通常是102
    tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids('[CLS]')  # 通常是101

    generator.config.pad_token_id = tokenizer.pad_token_id

    generator.resize_token_embeddings(len(tokenizer))
    discriminator_Y.resize_token_embeddings(len(tokenizer))

    return generator, discriminator_Y, discriminator_Z, intermediate_dis_Z, tokenizer


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
        
        return tf.cond(step < self.warmup_steps, warmup, decay)

def setup_optimizers():
    """
    we use a polynomial decay learning rate schedule with warmup here.
    """
    initial_learning_rate = 1e-5
    warmup_steps = 5 # 5%
    decay_steps = 95
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

    gen_optimizer = LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=final_lr_schedule), dynamic=True)
    dis_Y_optimizer = LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=1e-5), dynamic=True)
    dis_Z_optimizer = LossScaleOptimizer(tf.keras.optimizers.Adam(learning_rate=1e-5), dynamic=True)

    return gen_optimizer, dis_Y_optimizer, dis_Z_optimizer, final_lr_schedule


# 自定义训练步骤类
"""
in this task, we only trained X->Y.
"""
class Trainstep:
    def __init__(self, gen, gen_optimizer, dis_Y, dis_Y_optimizer, dis_Z, dis_Z_optimizer, intermediate_dis_Z, tokenizer, embedding, strategy, final_lr_schedule):
        self.strategy = strategy
        self.gen = gen
        self.gen_optimizer = gen_optimizer
        self.dis_Y = dis_Y
        self.dis_Y_optimizer = dis_Y_optimizer
        self.dis_Z = dis_Z
        self.dis_Z_optimizer = dis_Z_optimizer
        self.intermediate_model = intermediate_dis_Z
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.final_lr_schedule = final_lr_schedule
        self.accumulated_gradients = None


    def train_generator_step(self, input_ids, attention_mask, labels, styles, max_len, step, accumulation_steps=4, 
                             lambda_rec=1.0, lambda_lm=1.0, lambda_adv=1.0, lambda_kl=1.0, gamma=1.0): 
        
        max_len = tf.constant(max_len, dtype=tf.int32)
        max_len_value = max_len
        seq_len = input_ids.shape[2]
        max_new_tokens = tf.maximum(max_len_value - seq_len - 10, 1)
        max_new_tokens = tf.cast(max_new_tokens, tf.int32)
        max_new_tokens = tf.maximum(max_new_tokens, 1)
        max_new_tokens = tf.constant(max_new_tokens, dtype=tf.int32)

        """
        we firstly to perform data sharding to make sure that each device gets a batch of data.
        """
        input_ids, attention_mask, original_labels, original_styles = pr.distribute_data(input_ids, attention_mask, labels, styles)

        @tf.function
        def step_fn(input_ids, attention_mask, labels, styles, lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma):
            with tf.GradientTape() as tape:
                tf.debugging.enable_check_numerics()
                
                lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma = pr.conv_tensor_to_float(lambda_rec, lambda_lm, lambda_adv, lambda_kl, gamma)

                epsilon = 1e-6 # 快速修复

                replica_context = tf.distribute.get_replica_context()
                if replica_context:
                    replica_id = replica_context.replica_id_in_sync_group
                    if replica_id is not None:
                        print(f"Now is device {tf.get_static_value(replica_id)} training.")
                    else:
                        print("Training on an unknown device (replica_id is None).")
                else:
                    print("Training in a non-distributed context.")

                # 嵌入风格标签
                style_embeddings = self.embedding(styles) # [num_devices * batch_size, n_embd]
                print("Style embeddings shape:", style_embeddings.shape)  # Debug info
                
                # 将输入 ID 嵌入到相同的嵌入空间
                input_embeddings = self.gen.transformer.wte(input_ids) # [num_devices * batch_size, seq_len, n_embd]
                print("Input embeddings shape:", input_embeddings.shape)  # Debug info
                
                extended_input_embeddings = input_embeddings + tf.expand_dims(style_embeddings, axis=1)
                print("Extended embeddings shape:", extended_input_embeddings.shape)  # Debug info

                input_ids, attention_mask, labels, styles = dis.convert_tensor(input_ids, attention_mask, labels, styles)

                outputs = self.gen(input_ids=input_ids, attention_mask=attention_mask, training=True)
                logits = outputs.logits
                print("Logits shape:", logits.shape)  # Debug info
                print("Logits dtype:", logits.dtype)  # Debug info
                print("labels shape:", labels.shape)  # Debug info

                loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
                mask = tf.cast(labels != -100, logits.dtype)
                print("Mask shape:", mask.shape)  # Debug info
                print("Mask dtype:", mask.dtype)  # Debug info
        
                # Check for NaN or Inf in logits
                tf.debugging.check_numerics(logits, "Logits contain NaN or Inf")

                # 各个损失
                rec_loss = loss_fn(tf.where(labels == -100, tf.zeros_like(labels, dtype=logits.dtype), tf.cast(labels, logits.dtype)), logits)
                rec_loss = tf.reduce_sum(rec_loss * mask) / (tf.reduce_sum(mask) + epsilon)
                rec_loss = tf.cast(rec_loss, tf.float32)
                print("Reconstruction loss:", rec_loss)  # Debug info

                for var in self.gen.trainable_variables:
                    tf.debugging.check_numerics(var, message="Model weight check")

                print("Input shapes:", 
                        "input_ids:", input_ids.shape, 
                        "input_ids dtype:", input_ids.dtype,
                        "attention_mask:", attention_mask.shape, 
                        "labels:", labels.shape, 
                        "styles:", styles.shape,
                        "max_len_value:", max_len_value)

                new_shape = tf.shape(input_ids)
                print("New shape:", new_shape)  # Debug info
                print("Seq len:", seq_len)  # Debug info
                
                # max_new_tokens = tf.maximum(max_new_tokens, 1)
                print("Max length:", max_len_value)  # Debug info
                print("Max new tokens:", max_new_tokens)  # Debug info
                if isinstance(max_new_tokens, tf.Tensor):
                    print("Max new tokens:", tf.get_static_value(max_new_tokens))  # Debug info
                batch_size = new_shape[0]

                # 扩展 input_ids
                """
                at here, we need to padding to -> [batch_size, max_new_tokens]
                """
                padding = tf.zeros((batch_size, max_new_tokens), dtype=input_ids.dtype)
                print("Padding shape:", padding.shape)  # Debug info
                
                extended_input_ids = tf.concat([input_ids, padding], axis=1)
                extended_attention_mask1 = tf.concat([attention_mask, tf.zeros((tf.shape(attention_mask)[0],
                                            max_new_tokens), dtype=attention_mask.dtype)], axis=1)
                
                extended_input_ids = tf.cast(extended_input_ids, tf.int32)
                extended_attention_mask1 = tf.cast(extended_attention_mask1, tf.float32)
                
                print(f"Extended input_ids shape: {extended_input_ids.shape}")
                print(f"Extended attention_mask shape: {extended_attention_mask1.shape}")

                # 确保最大长度大于最小长度
                max_length = max_len_value + max_new_tokens
                min_length = 1
                # tf.print("max_length:", max_length)
                # tf.print("min_length:", min_length)

                tf.debugging.assert_greater(max_length, min_length, message=f"max_length ({max_length}) must be greater than min_length ({min_length})")
                
                pad_token_id = int(self.tokenizer.pad_token_id)
                eos_token_id = int(self.tokenizer.eos_token_id)
                bos_token_id = int(self.tokenizer.bos_token_id)

                try:                    
                    generated_ids = self.gen.generate(
                        extended_input_ids, 
                        attention_mask=extended_attention_mask1, 
                        max_new_tokens=50,
                        pad_token_id=pad_token_id,
                        eos_token_id=eos_token_id,
                        bos_token_id=bos_token_id,
                        # use_cache=True,
                        # num_beams=1,  # 使用贪婪搜索
                        do_sample=False,  # 不使用采样
                        # temperature=1.0,  # 降低随机性
                    )
                    print("Generation successful. Generated IDs shape:", generated_ids.shape)
                    print("Generated IDs dtype:", generated_ids.dtype)

                except Exception as e:
                    print(f"Error during generation: {e}")
                    print(f"input_ids shape: {input_ids.shape}")
                    print(f"attention_mask shape: {attention_mask.shape}")
                    print(f"max_len_value: {max_len_value}")
                    raise
                
                generated_embeddings = self.gen.transformer.wte(generated_ids) # [batch_size, seq_len, n_embd]
                extended_generated_embeddings = generated_embeddings + tf.expand_dims(style_embeddings, axis=1)
                print("Generated extended embeddings shape:", extended_generated_embeddings.shape)  # Debug info

                padded_input_ids = tf.pad(input_ids, [[0, 0], [0, tf.shape(generated_ids)[1] - tf.shape(input_ids)[1]]],
                                        "CONSTANT", constant_values=self.tokenizer.pad_token_id)
                zx_distribution = dis.compute_distribution(extended_input_embeddings, self.intermediate_model)
                zy_distribution = dis.compute_distribution(extended_generated_embeddings, self.intermediate_model)

                kl_loss = dis.kl_divergence(zx_distribution, zy_distribution)

                kl_loss = tf.cast(kl_loss, tf.float32)
                print("KL loss:", kl_loss)  # Debug info

                extended_attention_mask2 = tf.pad(attention_mask, [[0, 0],[0, tf.shape(generated_ids)[1] - tf.shape(attention_mask)[1]]], 
                                            "CONSTANT", constant_values=1)
                real_lm_loss_Y = dis.compute_lm_loss(generated_ids, extended_attention_mask2, self.dis_Y)
                real_lm_loss_Y = tf.cast(real_lm_loss_Y, tf.float32)
                lm_loss = gamma * real_lm_loss_Y
                print("LM loss:", lm_loss)  # Debug info

                real_adv_loss = self.dis_Y(padded_input_ids, attention_mask=extended_attention_mask2, training=True)
                # print("Real adv loss:", real_adv_loss)  # Debug info
                generated_adv_loss = self.dis_Y(generated_ids, attention_mask=extended_attention_mask2, training=True)

                real_adv_loss_logits = tf.clip_by_value(tf.cast(real_adv_loss.logits, tf.float32), -1e8, 1e8)
                generated_adv_loss_logits = tf.clip_by_value(tf.cast(generated_adv_loss.logits, tf.float32), -1e8, 1e8)

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
            
            print("Generator gradients calculated")  # Debug info
            return total_loss, rec_loss, lm_loss, adv_loss, kl_loss, gradients, accuracy

        
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in self.gen.trainable_variables]
        total_loss = tf.constant(0.0, dtype=tf.float32)
        total_rec_loss = tf.constant(0.0, dtype=tf.float32)
        total_lm_loss = tf.constant(0.0, dtype=tf.float32)
        total_adv_loss = tf.constant(0.0, dtype=tf.float32)
        total_kl_loss = tf.constant(0.0, dtype=tf.float32)
        total_accuracy = tf.constant(0.0, dtype=tf.float32)
        
        for account_steps in range(accumulation_steps):
            print(f"distribute shape: {tf.shape(input_ids)}, {tf.shape(attention_mask)}, {tf.shape(original_labels)}, {tf.shape(original_styles)}")
            print(f"accumulation distribute: {input_ids}, {attention_mask}, {original_labels}, {original_styles}")
            ids, mask, labels, styles = pr.accumulation_distribute(accumulation_steps, input_ids, attention_mask, original_labels, original_styles, account_steps)
            step_total_loss, step_rec_loss, step_lm_loss, step_adv_loss, step_kl_loss, step_gradients, step_accuracy = step_fn(
                ids, 
                mask, 
                labels, 
                styles, 
                lambda_rec, 
                lambda_lm, 
                lambda_adv, 
                lambda_kl, 
                gamma
            )
            print(f"accumulation_steps {account_steps + 1} finished.")
            total_loss += tf.cast(step_total_loss,tf.float32)
            total_rec_loss += tf.cast(step_rec_loss, tf.float32)
            total_lm_loss += tf.cast(step_lm_loss, tf.float32)
            total_adv_loss += tf.cast(step_adv_loss, tf.float32)
            total_kl_loss += tf.cast(step_kl_loss, tf.float32)
            total_accuracy += step_accuracy
            for i, g in enumerate(step_gradients):
                self.accumulated_gradients[i].assign_add(g)
        
        for i, g in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))
        current_lr = self.final_lr_schedule(step)
        
        # 在重置前拷贝一份梯度
        returned_gradients = [tf.identity(g) for g in self.accumulated_gradients]

        self.accumulated_gradients = None

        return (total_loss / tf.cast(accumulation_steps, tf.float32),
                total_rec_loss / tf.cast(accumulation_steps, tf.float32),
                total_lm_loss / tf.cast(accumulation_steps, tf.float32),
                total_adv_loss / tf.cast(accumulation_steps, tf.float32),
                total_kl_loss / tf.cast(accumulation_steps, tf.float32),
                current_lr,
                total_accuracy / tf.cast(accumulation_steps, tf.float32),
                returned_gradients)


    # REINFORCE算法
    def reinforce_step(self, input_ids, attention_mask, labels, styles, max_len, accumulation_steps=4):
        """
        we firstly to perform data sharding to make sure that each device gets a batch of data.
        """
        input_ids, attention_mask, origin_labels, origin_styles = pr.distribute_data(input_ids, attention_mask, labels, styles)

        @tf.function
        def step_fn(input_ids, attention_mask, labels, styles, max_len):
            with tf.GradientTape() as tape:
                # Debug first
                print("input_ids shape:", input_ids.shape)
                print("attention_mask shape:", attention_mask.shape)
                print("styles shape:", styles.shape)
                styles = tf.expand_dims(styles, axis=1)
                print("Max_length:", max_len)
                print("Fixed max length:", tf.get_static_value(max_len))

                generated_ids = self.gen.generate(input_ids, attention_mask=attention_mask, max_new_tokens=50)
                print("Generated IDs shape:", generated_ids.shape)
                generated_attention_mask = tf.pad(attention_mask, [[0, 0], [0, generated_ids.shape[1] - attention_mask.shape[1]]], "CONSTANT", constant_values=1)
                lm_loss = dis.compute_lm_loss(generated_ids, generated_attention_mask, self.dis_Y)
                lm_loss = tf.cast(lm_loss, tf.float32)
                
                outputs = self.gen(input_ids, attention_mask=attention_mask, labels=labels, training=False)
                logits = tf.cast(outputs.logits, tf.float32)
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
            print("Reinforce gradients calculated")  # Debug info
            return loss, gradients
        

        if self.accumulated_gradients is None:
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in self.gen.trainable_variables]
        total_loss = tf.constant(0.0, dtype=tf.float32)

        for account_step in range(accumulation_steps):
            ids, mask, labels, styles = pr.accumulation_distribute(accumulation_steps, input_ids, attention_mask, origin_labels, origin_styles, account_step)
            loss, gradients = step_fn(
                ids,
                mask,
                labels,
                styles,
                max_len
            )

            print(f"accumulation_steps {account_step + 1} finished.")
            total_loss += tf.cast(loss, tf.float32)
            for i, g in enumerate(gradients):
                self.accumulated_gradients[i].assign_add(g)

        for i, g in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))

        returned_gradients = [tf.identity(g) for g in self.accumulated_gradients]
        self.accumulated_gradients = None

        return loss, returned_gradients


    # 自定义鉴别器训练步骤
    def train_discriminator_Y_step(self, real_ids, real_mask, input_ids, attention_mask, max_len, accumulation_steps=4, gamma=1.0):
        """
        we firstly to perform data sharding to make sure that each device gets a batch of data.
        """
        real_ids, real_mask, predict_ids, predict_mask = pr.distribute_data_disc_Y(real_ids, real_mask, input_ids, attention_mask)

        @tf.function
        def step_fn(real_ids, real_mask, predict_ids, predict_mask):
            generated_ids = self.gen.generate(predict_ids, attention_mask=predict_mask, max_new_tokens=50)
            print(f"generated_ids.shape {generated_ids.shape} and real_ids.shape {real_ids.shape} and real_mask.shape {real_mask.shape}")
            with tf.GradientTape() as tape:
                real_loss = dis.compute_lm_loss(real_ids, real_mask, self.dis_Y)
                real_loss = tf.cast(real_loss, tf.float32)
                generated_mask = tf.pad(real_mask, [[0, 0], [0, generated_ids.shape[1] - real_ids.shape[1]]], "CONSTANT", constant_values=1)
                fake_loss = dis.compute_lm_loss(generated_ids, generated_mask, self.dis_Y)
                fake_loss = tf.cast(fake_loss, tf.float32)
                total_loss = real_loss - gamma * fake_loss
            gradients = tape.gradient(total_loss, self.dis_Y.trainable_variables)
            print("Discriminator Y gradients calculated")  # Debug info
            return total_loss, gradients


        if self.accumulated_gradients is None:
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in self.dis_Y.trainable_variables]
        total_loss = tf.constant(0.0, dtype=tf.float32)

        for account_step in range(accumulation_steps):
            Y_ids, Y_mask, X_ids, X_mask = pr.accumulation_distribute_disc_Y(accumulation_steps, real_ids, real_mask, predict_ids, predict_mask, account_step)
            loss, gradients = step_fn(
                Y_ids, 
                Y_mask, 
                X_ids, 
                X_mask
            )
            print(f"accumulation_steps {account_step + 1} finished.")
            total_loss += tf.cast(loss, tf.float32)
            for i, g in enumerate(gradients):
                self.accumulated_gradients[i].assign_add(g)
        
        for i, g in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))

        returned_gradients = [tf.identity(g) for g in self.accumulated_gradients]
        self.accumulated_gradients = None

        return total_loss, returned_gradients


    # 鉴别器z对齐
    @tf.function
    def discriminator_Z_loss(self, zx, zy, label_smoothing=0.1):
        label_smoothing = tf.cast(label_smoothing, tf.float32)
        
        zx = tf.cast(zx, tf.float32)
        zy = tf.cast(zy, tf.float32)

        zx = tf.cast(self.gen.transformer.wte(zx), tf.float32)
        zy = tf.cast(self.dis_Y.transformer.wte(zy), tf.float32)

        zx_logits = tf.cast(self.dis_Z(zx), tf.float32)
        zy_logits = tf.cast(self.dis_Z(zy), tf.float32)

        zx_pred = tf.nn.sigmoid(zx_logits)
        zy_pred = tf.nn.sigmoid(zy_logits)

        zx_label = tf.zeros_like(zx_pred, dtype=tf.float32)
        zy_label = tf.ones_like(zy_pred, dtype=tf.float32)

        # 自定义二元交叉熵损失
        def binary_crossentropy(y_true, y_pred, label_smoothing=0.0):
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
            return -tf.reduce_mean(y_true * tf.math.log(y_pred + 1e-7) + (1 - y_true) * tf.math.log(1 - y_pred + 1e-7))

        zx_loss = binary_crossentropy(zx_label, zx_pred, label_smoothing=label_smoothing)
        zy_loss = binary_crossentropy(zy_label, zy_pred, label_smoothing=label_smoothing)

        total_loss = tf.clip_by_value(zx_loss + zy_loss, -1e8, 1e8)
        return tf.cast(total_loss, tf.float32)


    # 训练鉴别器z
    @tf.function
    def train_discriminator_Z_step(self, zx, zy, label_smoothing=0.1):
        zx = tf.cast(tf.pad(zx, [[0, 0], [0, zy.shape[1] - zx.shape[1]]], "CONSTANT", constant_values=0), tf.float32)
        zy = tf.cast(zy, tf.float32)
        with tf.GradientTape() as tape:
            loss = tf.cast(self.discriminator_Z_loss(zx, zy, label_smoothing=label_smoothing), tf.float32)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.dis_Z.trainable_variables)
        print("Loss dtype:", loss.dtype)  # Debug info
        print("Discriminator Z gradients calculated")  # Debug info
        gradients, _ = tf.clip_by_global_norm(gradients, 0.9)
        return loss, gradients


    # 生成样本训练鉴别器z
    def train_generator_with_discriminator_Z(self, input_ids_X, attention_mask_X, style_ids_X, accumulation_steps=4, label_smoothing=0.1):
        style_ids_X = tf.cast(tf.expand_dims(style_ids_X, axis=1), tf.int32)

        """
        perform data sharding to make sure that each device gets a batch of data.
        """
        input_ids_X, attention_mask_X = pr.distribute_data_disc_Z(input_ids_X, attention_mask_X) 

        @tf.function
        def step_fn(input_ids_X, attention_mask_X):
            generated_ids_X_Z = tf.cast(self.gen.generate(input_ids_X, attention_mask=attention_mask_X, max_new_tokens=50), tf.int32)
            
            disc_z_loss, gradients = self.train_discriminator_Z_step(input_ids_X, generated_ids_X_Z, label_smoothing=label_smoothing)
            disc_z_loss = tf.cast(disc_z_loss, tf.float32)
            return tf.cast(disc_z_loss, tf.float32), gradients
        
        if self.accumulated_gradients is None:
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(g), trainable=False) for g in self.dis_Z.trainable_variables]
        total_loss = tf.constant(0.0, dtype=tf.float32)

        for account_step in range(accumulation_steps):
            ids_X, mask_X = pr.accumulation_distribute_disc_Z(accumulation_steps, input_ids_X, attention_mask_X, account_step)
            loss, gradients = step_fn(
                ids_X, 
                mask_X
            )
            print(f"accumulation_steps {account_step + 1} finished.")
            total_loss += tf.cast(loss, tf.float32)
            for i, g in enumerate(gradients):
                self.accumulated_gradients[i].assign_add(g)

        for i, g in enumerate(self.accumulated_gradients):
            self.accumulated_gradients[i].assign(g / tf.cast(accumulation_steps, g.dtype))

        returned_gradients = [tf.identity(g) for g in self.accumulated_gradients]
        self.accumulated_gradients = None

        return total_loss, returned_gradients
