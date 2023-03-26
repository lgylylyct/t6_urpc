from Trainer.args.defult import args


# supervised
args.exp = "SemiSup_SwinUNETR_T2"
args.semi_sup = True

# model
args.model = "SwinUNETR"

# dataset
args.mod = "T2"
args.label_mod_type = "T2_mask"


