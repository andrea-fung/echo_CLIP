def get_config():
    config = {}

    ###general
    #experiment name ##TODO
    config["exp_name"] = 'Debug'

    #Type of model (e.g. RN50, ViT-B/32) ##TODO
    config["model"] = None

    #Checkpoint of the model you want to test -##TODO
    config["test_comprehensive"] = "/workspace/echo_CLIP/logs/finetuned_endtoend/eclip_pretrain_finetuned_5/checkpoint_8.pt"
    config["dataset_root"] = "/workspace/data/as_tom"
    config["img_path_dataset"] = '/workspace/data/as_tom/annotations-all.csv'
    config["report_path_dataset"] = '/workspace/data/as_tom/finetuned_df_report_text.csv'

    #Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both
    config["load"] = None #"/workspace/echo_CLIP/logs/pretrained/eclip_pretrain_5/checkpoint_10.pt"
    #Optionally save a _classifier_, e.g. a zero shot classifier or probe
    config["finetune_save"] = "/workspace/echo_CLIP/logs/finetuned_endtoend"
    config["pretrain_save"] = "/workspace/echo_CLIP/logs/pretrained"
    config["results-db"] = "/workspace/echo_CLIP/logs"

    config["pretrain_epochs"] = 1 #40
    config["epochs"] = 20

    config["freeze-encoder"] = False
    config["use_wandb"] = False
    config["wandb_dir"] = "/workspace/echo_CLIP"

    config["mode"] = 'train'

    ###Dataloader
    #Number of workers for dataloader
    config["num_workers"] = 8
    config["sampler"] = 'random'
    

    ###Hyperparameters
    config["batch_size"] = 16
    config["lr"] = 1e-5
    
    config["view"] = 'all'
    config["flip_rate"] = 0.3
    config["label_scheme_name"] = 'all'
    # must be compatible with number of unique values in label scheme
    # will be automatic in future update
    # number of AS classes
    config["num_classes"] = 4
    # weight decay
    config["wd"] = 0.1
    # label smoothing
    config["ls"] = 0.0
    config["warmup_length"] = 500 #TODO

    config["video_input"] = True

    config["use_tab"] = True
    config["scale_feats"] = False
    config["drop_cols"] = []
    config["categorical_cols"] = []

    ###tufts
    config["tufts_droot"] = "/workspace/TMED/approved_users_only/"
    config["tufts_csv_name"] = 'DEV479/TMED2_fold0_labeledpart.csv'
    config["tufts_label_scheme_name"] = 'mod_severe'
    config["view_scheme_name"] = 'three_class'
    config["dataset"] = "tufts"

    ###Additional
    #Which class names to use. ##TODO
    config["classnames"] = "openai"
    #Directory for caching features and encoder
    config["cache-dir"] = None

    return config
