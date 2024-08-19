from transformers import BertTokenizer, TFGPT2LMHeadModel
import tensorflow as tf
import os
import experiment as ex
import discriminator as dis
import pre_progress as pr
import model as md
import logging

# 日志级别
logging.basicConfig(level=logging.INFO)

# 环境guarantee
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.pop('TF_CONFIG', None)


# os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices=false'
tf.config.optimizer.set_jit(False)  # 禁用 XLA

# 日志调试
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 紧急执行
# tf.config.run_functions_eagerly(True)

# 混合精度
from tensorflow.keras.mixed_precision import set_global_policy
# set_global_policy('mixed_float16')


# 获取当前文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path_X = os.path.join(current_dir, './corpus/X_corpus.txt')
file_path_Y = os.path.join(current_dir, './corpus/Y_corpus.txt')
test_file_path = os.path.join(current_dir, './corpus/test_corpus.txt')

accumulated_gradients = None


# # 检查点
# from multiprocessing import util
# checkpoint_dir = os.path.join(util.get_temp_dir(), 'ckpt')

# def _is_chief(task_type, task_id, cluster_spec):
#   return (task_type is None
#           or task_type == 'chief'
#           or (task_type == 'worker'
#               and task_id == 0
#               and "chief" not in cluster_spec.as_dict()))

# def _get_temp_dir(dirpath, task_id):
#   base_dirpath = 'workertemp_' + str(task_id)
#   temp_dir = os.path.join(dirpath, base_dirpath)
#   tf.io.gfile.makedirs(temp_dir)
#   return temp_dir

# def write_filepath(filepath, task_type, task_id, cluster_spec):
#   dirpath = os.path.dirname(filepath)
#   base = os.path.basename(filepath)
#   if not _is_chief(task_type, task_id, cluster_spec):
#     dirpath = _get_temp_dir(dirpath, task_id)
#   return os.path.join(dirpath, base)

# checkpoint = tf.train.Checkpoint(generator=generator, discriminator_Y=discriminator_Y, discriminator_Z=discriminator_Z)
# write_checkpoint_dir = write_filepath(checkpoint_dir, task_type, task_id, cluster_spec)
# checkpoint_manager = tf.train.CheckpointManager(checkpoint, directory=write_checkpoint_dir, max_to_keep=1)

# lastest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
# if lastest_checkpoint:
#     checkpoint.restore(lastest_checkpoint)


@tf.function 
def distributed_train_generator_step(self, **kwargs):
    per_replica_losses = self.strategy.run(md.train_generator_step, args=(**kwargs,))
    return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_train_discriminator_Y_step(strategy, dataset_inputs):
    per_replica_losses = strategy.run(md.train_discriminator_Y_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)


@tf.function
def distributed_train_discriminator_Z_step(strategy, dataset_inputs):
    per_replica_losses = strategy.run(md.train_discriminator_Z_step, args=(dataset_inputs,))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

# 训练
class Train(tf.keras.Model):
    def __init__(self, gen, dis_Y, dis_Z, config, strategy, mymodel):
        super(Train, self).__init__()
        self.cfg = config
        self.strategy = strategy
        self.mymodel=mymodel

        with self.strategy.scope():
            self.gen = gen
            self.dis_Y = dis_Y
            self.dis_Z = dis_Z
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

        self.final_lr_schedule = md.WarmUpDecay(
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
                        'valid_loss', 'valid_accuracy', 'perplexity', 'learning_rate']
        self.metrics = {name: tf.keras.metrics.Mean(name=name) for name in metric_names}


    def update_metrics(self, **kwargs) -> None:
        for name, value in kwargs:
            if name in self.metrics:
                self.metrics[name].update_state(value)


    def reset_metrics(self):
        for metric in self.metrics.values():
            metric.reset_states()

    @tf.function 
    def distributed_train_generator_step(self, **kwargs):
        gen_dict = self.strategy.run(md.train_generator_step, args=(**kwargs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, gen_dict, axis=None)


    @tf.function
    def distributed_train_discriminator_Y_step(self, dataset_inputs):
        dis_Y_dict = self.strategy.run(md.train_discriminator_Y_step, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, dis_Y_dict, axis=None)


    @tf.function
    def distributed_train_discriminator_Z_step(self, dataset_inputs):
        dis_Z_dict = self.strategy.run(md.train_discriminator_Z_step, args=(dataset_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, dis_Z_dict, axis=None)
    

    def train(self, train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, epochs, *args, **kwargs):
        train_dist_dataset_X = self.strategy.experimental_distribute_dataset(train_tf_dataset_X)
        train_dist_dataset_Y = self.strategy.experimental_distribute_dataset(train_tf_dataset_Y)
        valid_dist_dataset_X = self.strategy.experimental_distribute_dataset(valid_tf_dataset_X)

        for epoch in range(epochs):
            self.reset_metrics()

            for batch_X, batch_Y in zip(train_dist_dataset_X, train_dist_dataset_Y):
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
                            self.final_lr_schedule, self.mymodel, 
                            self.cfg.accumulation_steps,self.cfg.lambda_rec, 
                            self.cfg.lambda_lm, self.cfg.lambda_adv, 
                            self.cfg.lambda_kl, self.cfg.gamma)
                
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
                
                self.update_metrics(
                    train_loss=loss,
                    train_accuracy=accuracy,
                    rec_loss=rec_loss,
                    lm_loss=lm_loss,
                    adv_loss=adv_loss,
                    kl_loss=kl_loss,
                    disc_y_loss=disc_loss_Y,
                    disc_z_loss=(disc_z_loss_X + disc_z_loss_Y) / 2.0,
                    reinforce_loss=reinforce_loss,
                    learning_rate=current_lr
                )
                self.cfg.lr_step += 1
                print(f"Train Epoch {epoch + 1} completed")

            print(f"Valid Epoch {epoch + 1} started")
            for batch_valid_X in valid_dist_dataset_X:
                batch_valid_ids = batch_valid_X['input_ids']
                batch_valid_attention_mask = batch_valid_X['attention_mask']
                batch_valid_labels = ex.create_labels(batch_valid_ids, batch_valid_attention_mask)
                batch_valid_styles = batch_valid_X['style']

                batch_valid_attention_mask = tf.convert_to_tensor(batch_valid_attention_mask, dtype=tf.int32)
                valid_loss, valid_accuracy = ex.valid_step(self.gen, 
                                    batch_valid_ids, batch_valid_attention_mask, 
                                    batch_valid_labels, batch_valid_styles)

                self.update_metrics(
                    valid_loss=valid_loss,
                    valid_accuracy=valid_accuracy
                )

            self.metrics['valid_perplexity'] = tf.exp(self.metrics['valid_loss'].result())
            self.print_epoch_results()
            self.append_to_config()
            print(f"Valid Epoch {epoch + 1} completed")


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
        template = ''.join(f"{key}: {metric.result().numpy()}" for key, metric in self.metrics.items())
        print(template)


# 实例化并训练
import json

# 先设置环境变量
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        "chief": ["host1:2222"],
        "worker": ["host2:2222", "host3:2222", "host4:2222"]
    },
    "task": {"type": "worker", "index": 1}
})

communication_options = tf.distribute.experimental.CommunicationOptions(implementation=tf.distribute.experimental.CommunicationImplementation.NCCL)
mirrored_strategy = tf.distribute.MultiWorkerMirroredStrategy(communication_options=communication_options)

with mirrored_strategy.scope():
    generator, discriminator_Y, discriminator_Z, tokenizer = md.create_model()

    trainconfig = pr.Trainconfig()

    mymodel = ex.MyModel(generator)

    train_dataset_X, train_dataset_Y, valid_dataset_X, valid_dataset_Y, \
    test_dataset = pr.load_dataset(file_path_X, file_path_Y, test_file_path, tokenizer, seed=42)
    train_tf_dataset_X, train_tf_dataset_Y, valid_tf_dataset_X, valid_tf_dataset_Y, \
    test_tf_dataset = pr.create_tf_dataset(train_dataset_X, trainconfig.batch_size, shuffle=True), \
                      pr.create_tf_dataset(train_dataset_Y, trainconfig.batch_size, shuffle=True), \
                      pr.create_tf_dataset(valid_dataset_X, trainconfig.batch_size, shuffle=False), \
                      pr.create_tf_dataset(valid_dataset_Y, trainconfig.batch_size, shuffle=False), \
                      pr.create_tf_dataset(test_dataset, trainconfig.batch_size, shuffle=False)
    
    train_model = Train(generator, discriminator_Y, discriminator_Z, trainconfig, mirrored_strategy, mymodel)

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
