from Trainer.args.defult import args


# supervised
args.exp = "SemiSup_SwinUNETR_T1C_D6_W1"
args.semi_sup = True

# model
args.model = "SwinUNETR"

# dataset
args.mod = "T1C"
args.label_mod_type = "T1C_mask"
args.labeled_num = 6

# consistency
args.consistency = 1.0