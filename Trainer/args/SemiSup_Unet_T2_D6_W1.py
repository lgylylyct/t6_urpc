from Trainer.args.defult import args


# supervised
args.exp = "SemiSup_Unet_T2_D6_W1"
args.semi_sup = True

# model
args.model = "unet_urpc"

# dataset
args.mod = "T2"
args.label_mod_type = "T2_mask"
args.labeled_num = 6

# consistency
args.consistency = 1.0