from Trainer.args.defult import args


# supervised
args.exp = "SemiSup_SwinUNETR_T1C"
args.semi_sup = True

# model
args.model = "SwinUNETR"

# dataset
args.mod = "T1C"
args.label_mod_type = "T1C_mask"


