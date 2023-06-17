from Trainer.args.defult import args


# supervised
args.exp = ""
args.semi_sup = True

# model
args.model = "unet_urpc"
args.optim="adamw"

# dataset
args.mod = "T2"
args.label_mod_type = "T2_mask"
args.labeled_num = 6

# consistency
args.consistency = 1.0