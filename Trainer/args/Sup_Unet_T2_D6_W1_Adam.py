from Trainer.args.defult import args


# supervised
args.exp = "Sup_Unet_T2_D6_W1_Adam"
args.semi_sup = False

# model
args.model = "unet_urpc"
args.optim="adam"

# dataset
args.mod = "T2"
args.label_mod_type = "T2_mask"
args.labeled_num = 6

# consistency
args.consistency = 1.0