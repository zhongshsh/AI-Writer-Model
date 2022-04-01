import os, sys, time, math, random, json, datetime, logging, shutil
import numpy as np
import oneflow as flow
from oneflow.utils.data import Dataset
from src.trainer import Trainer, TrainerConfig
from src.model import GPT, GPTConfig
import src.utils
import config


src.utils.set_seed(42)
np.set_printoptions(precision=4, suppress=True, linewidth=200)

begin_time = time.strftime("%Y-%m-%d", time.localtime())
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    filename="./log/" + begin_time + "_" + config.DATA_NAME + ".log",
    filemode="w",
)
logger = logging.getLogger(__name__)


class DataHandler(Dataset):
    def __init__(
        self, data, model_level, model_save_path, ctx_len, epoch_length_fixed=(10000)
    ):

        print("building token list...", end=" ")

        self.epoch_length_fixed = epoch_length_fixed  # make an 'epoch' very short, so we can see the training progress

        if model_level == "word":
            import re

            data = re.sub(
                r"(\n|\.|\,|\?|\!|\:|\;|\-|\—|\||\'|\"|\`|\(|\)|[0-9]|\[|\]|\{|\}|\=|\+|\*|\\|\/|\~|\&|\$|\#|\%)",
                r" \g<0> ",
                data,
            )
            data = re.sub(" +", " ", data)
            print("splitting token...")
            data = data.lower().split(" ")
        unique = sorted(list(set(data)))

        xx = 0
        xxObj = {}
        for u in unique:
            xxObj[xx] = u
            xx += 1
        with open(model_save_path + "vocab.json", "w", encoding="utf-16") as vocab_file:
            vocab_file.write(json.dumps(xxObj, ensure_ascii=False))

        data_size, vocab_size = len(data), len(unique)
        print("data has %d %ss, %d unique." % (data_size, model_level, vocab_size))
        logger.info(
            "data has {} {}s, {} unique.".format(data_size, model_level, vocab_size)
        )
        self.stoi = {ch: i for i, ch in enumerate(unique)}
        self.itos = {i: ch for i, ch in enumerate(unique)}
        self.ctx_len = config.CTX_LEN
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return self.epoch_length_fixed

    def __getitem__(self, idx):
        i = np.random.randint(
            0, len(self.data) - (self.ctx_len + 1)
        )  # cheat: pick a random spot in dataset
        chunk = self.data[i : i + self.ctx_len + 1]
        dix = [self.stoi[s] for s in chunk]
        x = flow.tensor(dix[:-1], dtype=flow.long)
        y = flow.tensor(dix[1:], dtype=flow.long)
        return x, y


class ModelHandler:
    def __init__(self, model_type="RWKV"):

        self.model_save_path = "./model/{}/".format(config.DATA_NAME)

        if os.path.exists(self.model_save_path):
            shutil.rmtree(self.model_save_path)
        os.makedirs(self.model_save_path)
        self.load_data()

        n_embd = config.N_HEAD * 64
        n_attn = n_embd
        n_ffn = n_embd

        ######## special hyperparameters for RWKV model ########
        rwkv_emb_scale = 0.4  # scale of initial embedding. 0.4 is a good choice
        rwkv_tiny_attn = 0  # 64 if (datafile_type == 0 and config.CTX_LEN > 600) else 0 # extra tiny attention dim, useful for long ctx char-level english
        rwkv_tiny_head = 1  # 1 is good enough. 8 is slow

        self.model = GPT(
            GPTConfig(
                self.train_dataset.vocab_size,
                config.CTX_LEN,
                model_type=model_type,
                rwkv_emb_scale=rwkv_emb_scale,
                rwkv_tiny_attn=rwkv_tiny_attn,
                rwkv_tiny_head=rwkv_tiny_head,
                n_layer=config.N_LAYER,
                n_head=config.N_HEAD,
                n_embd=n_embd,
                n_attn=n_attn,
                n_ffn=n_ffn,
            )
        )

        if config.TRAINED_MODEL != "":
            # load a trained model
            print(f"loading model for {config.RUN_DEVICE}...")
            if config.RUN_DEVICE == "dml":
                import onnxruntime as rt

                sess_options = rt.SessionOptions()
                sess_options.graph_optimization_level = (
                    rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                sess_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
                sess_options.enable_mem_pattern = False
                rt_session = rt.InferenceSession(
                    config.MODEL_NAME + ".onnx",
                    sess_options=sess_options,
                    providers=["DmlExecutionProvider"],
                )
                rt_session.set_providers(["DmlExecutionProvider"])
            else:
                self.model.load_state_dict(flow.load(config.TRAINED_MODEL).state_dict())

            print("successfully loading model from ", config.TRAINED_MODEL)

        if config.RUN_DEVICE == "gpu":
            self.model = self.model.cuda()

        logger.info(str(config))

    def load_data(self):
        if config.DATA_NAME == "wangwen":
            with open(
                config.PTH_MODEL_PATH + ".json", "r", encoding="utf-16"
            ) as result_file:
                word_table = json.load(result_file)

            self.train_dataset = lambda: None
            self.train_dataset.stoi = {v: int(k) for k, v in word_table.items()}
            self.train_dataset.itos = {int(k): v for k, v in word_table.items()}
            self.train_dataset.vocab_size = len(word_table)

        else:
            data_path = "./data/{}/".format(config.DATA_NAME)
            datafile = data_path + "train.txt"
            datafile_encoding = "utf-8"
            datafile_type = 0  # use 0 for char-level english. use 1 for chinese. only affects some RWKV hyperparametrs

            print("data path: " + datafile)
            logger.info("data path: " + datafile)

            self.train_dataset = DataHandler(
                open(datafile, "r", encoding=datafile_encoding).read(),
                config.MODEL_LEVEL,
                self.model_save_path,
                config.CTX_LEN,
            )
            self.vocab_size = self.train_dataset.vocab_size

    def train(self):
        weight_decay = (
            0 if config.MODEL_TYPE == "RWKV" else 0.01
        )  # wd is not useful when we have enough data
        lr_init = (
            6e-4 if config.MODEL_TYPE == "RWKV" else 4e-4
        )  # RWKV can use higher lr.  8e-4 = 0.0008   4e-4 = 0.0004
        lr_final = 4e-5
        betas = (0.9, 0.99) if config.MODEL_TYPE == "RWKV" else (0.9, 0.99)
        eps = 4e-9
        epoch_save_path = self.model_save_path + "trained-"

        tconf = TrainerConfig(
            model_type=config.MODEL_TYPE,
            max_epochs=config.N_EPOCH,
            batch_size=config.BATCH_SIZE,
            weight_decay=weight_decay,
            learning_rate=lr_init,
            lr_decay=True,
            lr_final=lr_final,
            betas=betas,
            eps=eps,
            warmup_tokens=0,
            final_tokens=config.N_EPOCH * len(self.train_dataset) * config.CTX_LEN,
            num_workers=0,
            epoch_save_frequency=config.EPOCH_SAVE_FREQUENCY,
            epoch_save_path=epoch_save_path,
        )
        trainer = Trainer(self.model, self.train_dataset, None, tconf)

        trainer.train()
        save_path = self.model_save_path + trainer.get_run_name()
        flow.save(self.model, save_path)
        logger.info(save_path)
        print("save path: ", save_path)

    def test(self, s=config.CONTEXT):
        print("input context: ", s)
        UNKNOWN_CHAR = self.train_dataset.stoi["0"]
        for run in range(config.NUM_OF_RUN):
            x = np.array(
                [self.train_dataset.stoi.get(s, UNKNOWN_CHAR) for s in config.CONTEXT],
                dtype=np.int64,
            )

            real_len = len(x)
            print_begin = 0

            for i in range(config.LENGTH_OF_EACH):

                if i == 0:
                    print(
                        ("-" * 60)
                        + "\n"
                        + config.CONTEXT.replace("\n", "\n  ").strip("\n"),
                        end="",
                    )
                    print_begin = real_len

                with flow.no_grad():
                    if config.RUN_DEVICE == "dml":
                        if real_len < config.CTX_LEN:
                            xxx = np.pad(x, (0, config.CTX_LEN - real_len))
                        else:
                            xxx = x
                        out = rt_session.run(
                            None,
                            {rt_session.get_inputs()[0].name: [xxx[-config.CTX_LEN :]]},
                        )
                        out = flow.tensor(out[0])
                    else:
                        xxx = flow.tensor(x[-config.CTX_LEN :], dtype=flow.long)[
                            None, ...
                        ]

                        if config.RUN_DEVICE == "gpu":
                            xxx = xxx.cuda()
                        out, _ = self.model(xxx)
                    out[:, :, UNKNOWN_CHAR] = -float("Inf")
                pos = -1 if real_len >= config.CTX_LEN else real_len - 1

                if self.train_dataset.itos[int(x[real_len - 1])] == "\n":
                    char = src.utils.sample_logits(
                        out, pos, temperature=1.0, top_p=config.TOP_P_NEWLINE
                    )
                else:
                    char = src.utils.sample_logits(
                        out, pos, temperature=1.0, top_p=config.TOP_P
                    )

                x = np.append(x, char)
                real_len += 1

                if (
                    i % 2 == 1
                    or i == config.LENGTH_OF_EACH - 1
                    or i < 10
                    or config.RUN_DEVICE != "gpu"
                ):
                    completion = "".join(
                        [
                            self.train_dataset.itos[int(i)]
                            for i in x[print_begin:real_len]
                        ]
                    )
                    print(completion.replace("\n", "\n  "), end="", flush=True)
                    print_begin = real_len

            print()

    def convert_pth_to_oneflow(
        self, pth_model_path=config.PTH_MODEL_PATH, save_path=""
    ):
        import torch

        parameters = torch.load(
            pth_model_path + ".pth", map_location="cpu"
        ).state_dict()
        for key, value in parameters.items():
            val = value.detach().cpu().numpy()
            parameters[key] = val
        self.model.load_state_dict(parameters)

        if save_path != "":
            flow.save(model, save_path)

        print("successfully convert pth model: ", config.PTH_MODEL_PATH)


if __name__ == "__main__":

    main = ModelHandler()
    if config.TASK == "train":
        main.train()
    elif config.TASK == "test":
        main.test()
    elif config.TASK == "convert_pth_to_oneflow":
        main.convert_pth_to_oneflow()
    else:
        print(
            """
Your config of TASK is {}. 
- if pretrained model comes from torch, you can use `config.TASK = convert_pth_to_oneflow` to load it. 
- if you want to test the trained model, you can use `config.TASK = test` to see the text prediction. 
- if you want to train a model, you can use `config.TASK = train`. Besides, you can change the value of config.TRAINED_MODEL to load a trained model.
            """.format(
                config.TASK
            )
        )
