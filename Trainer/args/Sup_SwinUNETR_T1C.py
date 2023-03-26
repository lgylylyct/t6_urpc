from Trainer.args.defult import args


# supervised
args.exp = "Sup_SwinUNETR_T1C"
args.semi_sup = False

# model
args.model = "SwinUNETR"

# dataset
args.mod = "T1C"
args.label_mod_type = "T1C_mask"


