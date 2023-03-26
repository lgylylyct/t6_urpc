from Trainer.args.defult import args


# supervised
args.exp = "Sup_Unet_T1C_D6_W1"
args.semi_sup = False

# model
args.model = "unet_urpc"

# dataset
args.mod = "T1C"
args.label_mod_type = "T1C_mask"
args.labeled_num = 6

# consistency
args.consistency = 1.0