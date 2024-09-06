from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import os
import experiment as ex
import discriminator as dis
import pre_progress as pr
import model as md
import numpy as np
import datetime

# 日志级别
# logging.basicConfig(level=logging.INFO)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# 环境guarantee
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# os.environ.pop('TF_CONFIG', None)

# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)  # 禁用 XLA

# 日志调试
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# 混合精度
from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')


# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_X = os.path.join(current_dir, './corpus/X_corpus.txt')
file_path_Y = os.path.join(current_dir, './corpus/Y_corpus.txt')
test_file_path = os.path.join(current_dir, './corpus/test_corpus.txt')

# TensorBoard
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = 'logs/' + current_time

tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch='2,5')


# 训练
class Train:
    def __init__(self, config, strategy, model):
        self.config = config
        self.strategy = strategy
        self.model = model
        self.setup_metrics()


    def setup_metrics(self):
        metric_names = ['train_loss', 'train_accuracy', 'rec_loss', 'lm_loss', 'adv_loss', 
                        'kl_loss', 'disc_y_loss', 'disc_z_loss', 'reinforce_loss', 
                        'valid_loss', 'valid_accuracy', 'valid_perplexity']
        self.metrics = {name: tf.keras.metrics.Mean(name=name) for name in metric_names}


    def update_metrics(self, **kwargs) -> None:
        def update_fn(metrics, **values):
            for name, value in values.items():
                if name in metrics:
                    if isinstance(value, tf.Tensor):
                        value = value.numpy()  
                    if np.isscalar(value):  
                        metrics[name].update_state(value)
                    else:
                        raise ValueError(f"Expected scalar value for metric '{name}', but got {type(value)}")

        self.strategy.run(update_fn, args=(self.metrics,), kwargs=kwargs)        


    def reset_metrics(self):
        for metric in self.metrics.values():
            if isinstance(metric, tf.keras.metrics.Metric):
                metric.reset_states()
            else:
                raise TypeError(f"Expected a tf.keras.metrics.Metric instance, but got {type(metric)}")


    def distributed_train_generator_step(self, batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, *args, **kwargs):
        new_args = []
        for arg in args:
            if isinstance(arg, tf.Tensor):
                new_args.append(arg)
            else:
                new_args.append(tf.convert_to_tensor(arg, dtype=tf.int32))
        
        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, tf.Tensor):
                new_kwargs[key] = value
            else:
                new_kwargs[key] = tf.convert_to_tensor(value)
        

        def generator_step(*args, **kwargs):
            loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy, gradients = self.model.train_generator_step(*args, **kwargs)
            self.model.gen_optimizer.apply_gradients(zip(gradients, self.model.gen.trainable_variables))
            return loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy

        # input_ids, attention_mask, labels, styles = pr.distribute_data(input_ids, attention_mask, labels, styles)
        loss, rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy = self.strategy.run(
            generator_step, 
            args=(batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, *new_args), 
            kwargs=new_kwargs
        )

        print("Generator graidents applied")
        total_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, loss, axis=None)
        rec_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, rec_loss, axis=None)
        lm_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, lm_loss, axis=None)
        adv_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, adv_loss, axis=None)
        kl_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, kl_loss, axis=None)
        current_lr = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, current_lr, axis=None)
        accuracy = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, accuracy, axis=None)
        return rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy, total_loss


    def distributed_train_reinforce_step(self, batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, *args, **kwargs):
        new_args = []
        for arg in args:
            if not isinstance(arg, tf.Tensor):
                arg = tf.convert_to_tensor(arg, dtype=tf.int32)
            new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, tf.Tensor):
                new_kwargs[key] = value
            else:
                new_kwargs[key] = tf.convert_to_tensor(value)


        def reinforce_step(*args, **kwargs):
            reinforce_loss, gradients = self.model.reinforce_step(*args, **kwargs)
            self.model.gen_optimizer.apply_gradients(zip(gradients, self.model.gen.trainable_variables))
            return reinforce_loss


        reinforce_loss = self.strategy.run(
            reinforce_step, 
            args=(batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X, *new_args), 
            kwargs=new_kwargs
        )
        print("Reinforce graidents applied")
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, reinforce_loss, axis=None)


    def distributed_train_discriminator_Y_step(self, batch_input_ids_Y, batch_attention_mask_Y, batch_input_ids_X, batch_attention_mask_X, *args, **kwargs):
        new_args = []
        for arg in args:
            if not isinstance(arg, tf.Tensor):
                arg = tf.convert_to_tensor(arg, dtype=tf.int32)
            new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, tf.Tensor):
                new_kwargs[key] = value
            else:
                new_kwargs[key] = tf.convert_to_tensor(value)


        def discriminator_Y_step(*args, **kwargs):
            reinforce_loss, gradients = self.model.train_discriminator_Y_step(*args, **kwargs)
            self.model.dis_Y_optimizer.apply_gradients(zip(gradients, self.model.dis_Y.trainable_variables))
            return reinforce_loss


        disc_loss_Y = self.strategy.run(
            discriminator_Y_step, 
            args=(batch_input_ids_Y, batch_attention_mask_Y, batch_input_ids_X, batch_attention_mask_X, *new_args), 
            kwargs=new_kwargs
        )
        print("Discriminator Y graidents applied")
        total_loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_loss_Y, axis=None)
        return total_loss


    def distributed_train_discriminator_Z_step(self, batch_input_ids_X, batch_attention_mask_X, batch_styles_X, *args, **kwargs):
        new_args = []
        for arg in args:
            if not isinstance(arg, tf.Tensor):
                arg = tf.convert_to_tensor(arg, dtype=tf.int32)
            new_args.append(arg)

        new_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, tf.Tensor):
                new_kwargs[key] = value
            else:
                new_kwargs[key] = tf.convert_to_tensor(value)


        def discriminator_Z_step(*args, **kwargs):
            disc_z_loss, gradients = self.model.train_generator_with_discriminator_Z(*args, **kwargs)

            tf.print("disc_z_loss:", disc_z_loss)
            tf.print("gradients length:", len(gradients))
            for i, grad in enumerate(gradients):
                if grad is not None:
                    tf.print(f"gradient {i} stats:", 
                            "shape:", tf.shape(grad), 
                            "dtype:", grad.dtype, 
                            "mean:", tf.reduce_mean(grad), 
                            "max:", tf.reduce_max(grad), 
                            "min:", tf.reduce_min(grad))
                else:
                    tf.print(f"gradient {i} is None")

            self.model.dis_Z_optimizer.apply_gradients(zip(gradients, self.model.dis_Z.trainable_variables))
            return disc_z_loss


        disc_z_loss = self.strategy.run(
            discriminator_Z_step, 
            args=(batch_input_ids_X, batch_attention_mask_X, batch_styles_X, *new_args), 
            kwargs=new_kwargs
        )
        print("Discriminator_Z graidents applied")
        loss = self.strategy.reduce(tf.distribute.ReduceOp.MEAN, disc_z_loss, axis=None)
        return loss
    
    def print_labels(self, train_batch, labels):
        def print_(train_batch, labels):
            if (train_batch == 0):
                print("labels:", labels)
        self.strategy.run(print_, args=(train_batch, labels))

    def train(self, train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, epochs, *args, **kwargs):
        train_dist_dataset_X = self.strategy.experimental_distribute_dataset(train_tf_dataset_X)
        train_dist_dataset_Y = self.strategy.experimental_distribute_dataset(train_tf_dataset_Y)

        """
        for validate, we donnot need to distribute the dataset
        """
        for epoch in range(epochs):
            self.reset_metrics()
            train_batch = 0

            for batch_X, batch_Y in zip(train_dist_dataset_X, train_dist_dataset_Y):
                print(f"Train Epoch {epoch + 1} started, batch {train_batch + 1}")

                batch_input_ids_X, batch_attention_mask_X, batch_labels_X, batch_styles_X = pr.get_params(batch_X)
                batch_input_ids_Y, batch_attention_mask_Y, batch_labels_Y, batch_styles_Y = pr.get_params(batch_Y)
               
                """
                we got the PerReplica object from the dataset
                """
                print("Processing batch")

                # 动态 Padding
                max_len_X, batch_input_ids_X, batch_attention_mask_X = dis.dynamic_padding(batch_input_ids_X, batch_attention_mask_X)
                max_len_Y, batch_input_ids_Y, batch_attention_mask_Y = dis.dynamic_padding(batch_input_ids_Y, batch_attention_mask_Y)
                max_len = tf.reduce_max(tf.stack([max_len_X, max_len_Y])) + 51
                print("max_len:", max_len)

                batch_input_ids_X = tf.convert_to_tensor(batch_input_ids_X, dtype=tf.int32)
                batch_attention_mask_X = tf.convert_to_tensor(batch_attention_mask_X, dtype=tf.int32)
                batch_input_ids_Y = tf.convert_to_tensor(batch_input_ids_Y, dtype=tf.int32)
                batch_attention_mask_Y = tf.convert_to_tensor(batch_attention_mask_Y, dtype=tf.int32)

                print("batch_input_ids_X shape:", batch_input_ids_X.shape)
                print("batch_attention_mask_X shape:", batch_attention_mask_X.shape)
                print("batch_input_ids_Y shape:", batch_input_ids_Y.shape)
                print("batch_attention_mask_Y.shape:", batch_attention_mask_Y.shape)

                # print("batch input_ids X:", batch_input_ids_X)

                tf.profiler.experimental.start(log_dir)
                
                print("Training gen")
                with tf.profiler.experimental.Trace('train_generator_step', step_num=train_batch, _r=1):
                    rec_loss, lm_loss, adv_loss, kl_loss, current_lr, accuracy, total_gen_loss = self.distributed_train_generator_step(
                        batch_input_ids_X,
                        batch_attention_mask_X,
                        batch_labels_X,
                        batch_styles_X,
                        max_len, 
                        tf.cast(self.config.lr_step, tf.float32), 
                        accumulation_steps=self.config.accumulation_steps,
                        lambda_rec=self.config.lambda_rec, 
                        lambda_lm=self.config.lambda_lm, 
                        lambda_adv=self.config.lambda_adv, 
                        lambda_kl=self.config.lambda_kl, 
                        gamma=self.config.gamma
                    )

                tf.profiler.experimental.stop()
                
                print("Training REINFORCE")
                reinforce_loss = self.distributed_train_reinforce_step(
                    batch_input_ids_X, 
                    batch_attention_mask_X, 
                    batch_labels_X, 
                    batch_styles_X, 
                    max_len,
                    accumulation_steps=self.config.accumulation_steps
                )

                print("Training discriminator Y")
                total__Y_loss = self.distributed_train_discriminator_Y_step(
                    batch_input_ids_Y, 
                    batch_attention_mask_Y, 
                    batch_input_ids_X,
                    batch_attention_mask_X,
                    max_len, 
                    accumulation_steps=self.config.accumulation_steps,
                    gamma=self.config.gamma
                )
                
                print("Training discriminator Z")
                total_Z_loss = self.distributed_train_discriminator_Z_step(
                    batch_input_ids_X, 
                    batch_attention_mask_X, 
                    batch_styles_X, 
                    accumulation_steps=self.config.accumulation_steps,
                    label_smoothing=self.config.label_smoothing
                )

                self.update_metrics(
                    train_loss=total_gen_loss,
                    train_accuracy=accuracy,
                    rec_loss=rec_loss,
                    lm_loss=lm_loss,
                    adv_loss=adv_loss,
                    kl_loss=kl_loss,
                    disc_y_loss=total__Y_loss,
                    disc_z_loss=total_Z_loss,
                    reinforce_loss=reinforce_loss,
                )

                self.config.lr_step += 1
                print(f"Train Epoch {epoch + 1} completed")
                train_batch += 1
                self.append_to_config_lr(current_lr)

            self.validate(epoch, valid_tf_dataset_X)

            perplexity = tf.exp(self.metrics['valid_loss'].result())
            self.update_metrics(valid_perplexity=perplexity)
            self.print_epoch_results()
            self.append_to_config()
            print(f"Epoch {epoch + 1} completed")
            epoch += 1


    def validate(self, epoch, valid_tf_dataset_X):
        def val_step(batch):
            batch_valid_ids, batch_valid_attention_mask, batch_valid_labels, batch_valid_styles = ex.get_params(batch) # we use ex.py function

            batch_valid_ids = tf.convert_to_tensor(batch_valid_ids, dtype=tf.int32)
            batch_valid_attention_mask = tf.convert_to_tensor(batch_valid_attention_mask, dtype=tf.int32)

            valid_loss, valid_accuracy = ex.valid_step(
                self.model.gen, 
                self.model.embedding,
                batch_valid_ids, 
                batch_valid_attention_mask, 
                batch_valid_labels, 
                batch_valid_styles
            )

            return valid_loss, valid_accuracy

        def validate_on_chief(epoch, dataset):
            valid_batch = 0
            for batch_valid_X in dataset:
                print(f"Valid Epoch {epoch + 1} started, batch {valid_batch + 1}")

                valid_loss, valid_accuracy = val_step(batch_valid_X)

                tf.distribute.get_replica_context().merge_call(
                    lambda strategy: self.update_metrics_distributed(valid_loss, valid_accuracy)
                )

                print(f"Valid Epoch {epoch + 1}, batch {valid_batch + 1} completed")
                valid_batch += 1

            print(f"Valid Epoch {epoch + 1} completed")

        def distributed_validate(epoch, dataset):
            if tf.equal(tf.distribute.get_replica_context().replica_id_in_sync_group, 0):  # The master copy performs validation
                validate_on_chief(epoch, dataset)
            else:
                # Ensure all replicas call merge_call the same number of times
                for _ in dataset:
                    tf.distribute.get_replica_context().merge_call(lambda strategy: None)

        self.strategy.run(distributed_validate, args=(epoch, valid_tf_dataset_X))


    def update_metrics_distributed(self, valid_loss, valid_accuracy):
        def update_fn(valid_loss, valid_accuracy):
            if isinstance(valid_loss, tf.Tensor):
                valid_loss = valid_loss.numpy()
            if isinstance(valid_accuracy, tf.Tensor):
                valid_accuracy = valid_accuracy.numpy()
            self.metrics['valid_loss'].update_state(valid_loss)
            self.metrics['valid_accuracy'].update_state(valid_accuracy)

        self.strategy.run(update_fn, args=(valid_loss, valid_accuracy))


    def append_to_config(self):
        self.config.train_losses.append(self.metrics['train_loss'].result().numpy())
        self.config.train_accuracies.append(self.metrics['train_accuracy'].result().numpy())
        self.config.rec_losses.append(self.metrics['rec_loss'].result().numpy())
        self.config.lm_losses.append(self.metrics['lm_loss'].result().numpy())
        self.config.adv_losses.append(self.metrics['adv_loss'].result().numpy())
        self.config.kl_losses.append(self.metrics['kl_loss'].result().numpy())
        self.config.disc_y_losses.append(self.metrics['disc_y_loss'].result().numpy())
        self.config.disc_z_losses.append(self.metrics['disc_z_loss'].result().numpy())
        self.config.reinforce_losses.append(self.metrics['reinforce_loss'].result().numpy())
        self.config.valid_losses.append(self.metrics['valid_loss'].result().numpy())
        self.config.valid_accuracies.append(self.metrics['valid_accuracy'].result().numpy())
        self.config.perplexities.append(self.metrics['valid_perplexity'].result().numpy())
    

    def append_to_config_lr(self, current_lr):
        if not isinstance(current_lr, float):
            current_lr = current_lr.numpy()
        self.config.learning_rates.append(current_lr)


    def print_epoch_results(self):
        def format_metric(metric):
            if hasattr(metric, 'result'):
                return metric.result().numpy()
            elif isinstance(metric, tf.Tensor):
                return metric.numpy()
            elif np.isscalar(metric):
                return metric
            else:
                return str(metric)
        template = ''.join(f"{key}: {format_metric(self.metrics[key])}\n" for key in self.metrics)
        print(template)


# 实例化并训练
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    generator, discriminator_Y, discriminator_Z, intermediate_model, tokenizer = md.create_model()

    trainconfig = pr.TrainConfig()

    embedding = ex.Embedding(generator)

    train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, test_dataset = pr.load_dataset(
        file_path_X, file_path_Y, test_file_path, tokenizer, seed=42
    )

    train_tf_dataset_X = pr.create_tf_dataset(train_dataset_X, trainconfig.batch_size)
    train_tf_dataset_Y = pr.create_tf_dataset(train_dataset_Y, trainconfig.batch_size)
    valid_tf_dataset_X = pr.create_tf_dataset(valid_dataset_X, trainconfig.batch_size)
    valid_tf_dataset_Y = pr.create_tf_dataset(valid_dataset_Y, trainconfig.batch_size)
    test_tf_dataset = pr.create_tf_dataset(test_dataset, trainconfig.batch_size)

    # 检查数据集
    for dataset_name, dataset in [("train_X", train_tf_dataset_X), ("train_Y", train_tf_dataset_Y), 
                              ("valid_X", valid_tf_dataset_X), ("valid_Y", valid_tf_dataset_Y)]:
        for batch in dataset.take(1):
            print(f"{dataset_name} shape:")
            print(f"  input_ids: {batch['input_ids'].shape}")
            print(f"  attention_mask: {batch['attention_mask'].shape}")
            print(f"  style: {batch['style'].shape}")

    gen_optimizer, dis_Y_optimizer, dis_Z_optimizer, final_lr_schedule = md.setup_optimizers()

    model = md.Trainstep(
        generator, 
        gen_optimizer, 
        discriminator_Y, 
        dis_Y_optimizer, 
        discriminator_Z, 
        dis_Z_optimizer,
        intermediate_model, 
        tokenizer, 
        embedding, 
        mirrored_strategy,
        final_lr_schedule
    )

    train_model = Train(trainconfig, mirrored_strategy, model)

    train_model.train(train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, trainconfig.epochs)

    # 测试集评估
    test_accuracies, test_losses, test_perplexity = ex.test_evalution(generator, ex.test_step, test_tf_dataset)

# 保存绘图
save_dir = './experiment'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
ex.perplexity_curve(train_model.config.perplexities, save_dir)
ex.loss_curve(train_model.config.train_losses, train_model.config.valid_losses, save_dir)
ex.accuracy_curve(train_model.config.train_accuracies, train_model.config.valid_accuracies, save_dir)
ex.learning_rate_curve(train_model.config.learning_rates, save_dir)
ex.plot_losses(train_model.config.rec_losses, train_model.config.lm_losses, train_model.config.adv_losses,
                train_model.config.kl_losses, train_model.config.disc_y_losses, train_model.config.disc_z_losses, save_dir)

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
