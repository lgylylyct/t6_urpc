from Trainer.args.defult import args


# supervised
args.exp = "Sup_SwinUNETR_T2"
args.semi_sup = False

# model
args.model = "SwinUNETR"

# dataset
args.mod = "T2"
args.label_mod_type = "T2_mask"


